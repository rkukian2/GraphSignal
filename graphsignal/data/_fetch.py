"""Smart fetcher utilities: typed errors, retry policy, circuit breaker.

Used by ingestion modules (prices, reference, events) to wrap network calls
to external data sources. Goals:

- Retry only transient failures (network blips, timeouts, rate limits, 5xx).
- Skip permanent failures (404, malformed payload, missing ticker) immediately.
- Halt the whole run after too many consecutive failures, instead of hammering
  a degraded upstream.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, TypeVar

import tenacity

from graphsignal.config import FetchConfig, get_config

log = logging.getLogger(__name__)

T = TypeVar("T")


class FetchError(Exception):
    """Base class for fetcher-layer errors."""


class TransientError(FetchError):
    """Retriable failure: network error, timeout, rate limit, 5xx."""


class PermanentError(FetchError):
    """Non-retriable failure: 404, malformed payload, missing ticker."""


class CircuitOpenError(FetchError):
    """Circuit breaker has tripped; the run should halt."""


class CircuitBreaker:
    """Trips after N consecutive failures; resets on any success.

    One breaker per ingestion run, shared across all fetches in that run.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.consecutive_failures = 0

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def check(self) -> None:
        if self.consecutive_failures >= self.threshold:
            raise CircuitOpenError(
                f"Circuit breaker open after {self.consecutive_failures} "
                f"consecutive failures (threshold={self.threshold})"
            )


def fetch_with_retry(
    fn: Callable[[], T],
    breaker: Optional[CircuitBreaker] = None,
    *,
    fetch_config: Optional[FetchConfig] = None,
    label: str = "fetch",
) -> T:
    """Run `fn` with smart retries.

    Retries only `TransientError`, with exponential backoff + jitter. Permanent
    errors are re-raised immediately. The circuit breaker (if provided) sees
    every outcome and aborts the run when consecutive failures pile up.
    """
    if breaker is not None:
        breaker.check()

    cfg = fetch_config or get_config().fetch

    retryer = tenacity.Retrying(
        stop=tenacity.stop_after_attempt(cfg.max_retries),
        wait=tenacity.wait_random_exponential(
            multiplier=cfg.initial_backoff_seconds,
            max=cfg.max_backoff_seconds,
        ),
        retry=tenacity.retry_if_exception_type(TransientError),
        before_sleep=tenacity.before_sleep_log(log, logging.WARNING),
        reraise=True,
    )

    try:
        result = retryer(fn)
    except FetchError:
        if breaker is not None:
            breaker.record_failure()
        raise
    if breaker is not None:
        breaker.record_success()
    return result
