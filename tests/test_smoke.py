"""Step 1 smoke tests: package imports, config loads, fetcher behaves correctly."""
import pytest

import graphsignal
from graphsignal.config import FetchConfig, GraphSignalConfig, get_config
from graphsignal.data._fetch import (
    CircuitBreaker,
    CircuitOpenError,
    PermanentError,
    TransientError,
    fetch_with_retry,
)

# Tiny config so retry tests run in milliseconds, not seconds.
FAST_FETCH = FetchConfig(
    max_retries=4,
    initial_backoff_seconds=0.001,
    max_backoff_seconds=0.005,
    concurrency=2,
    circuit_breaker_consecutive_failures=3,
)


def test_package_version():
    assert graphsignal.__version__


def test_default_config_loads():
    cfg = get_config()
    assert isinstance(cfg, GraphSignalConfig)
    assert cfg.fetch.max_retries >= 1
    assert cfg.events.drawdown_threshold_pct > 0


def test_circuit_breaker_trips_after_threshold():
    breaker = CircuitBreaker(threshold=3)
    for _ in range(3):
        breaker.record_failure()
    with pytest.raises(CircuitOpenError):
        breaker.check()


def test_circuit_breaker_resets_on_success():
    breaker = CircuitBreaker(threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    assert breaker.consecutive_failures == 0
    breaker.check()  # does not raise


def test_fetch_retries_transient_then_succeeds():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise TransientError("temporary glitch")
        return "ok"

    result = fetch_with_retry(flaky, fetch_config=FAST_FETCH)
    assert result == "ok"
    assert attempts["count"] == 3


def test_fetch_does_not_retry_permanent():
    attempts = {"count": 0}

    def fails():
        attempts["count"] += 1
        raise PermanentError("ticker not found")

    with pytest.raises(PermanentError):
        fetch_with_retry(fails, fetch_config=FAST_FETCH)
    assert attempts["count"] == 1


def test_fetch_gives_up_after_max_retries():
    attempts = {"count": 0}

    def always_transient():
        attempts["count"] += 1
        raise TransientError("upstream down")

    with pytest.raises(TransientError):
        fetch_with_retry(always_transient, fetch_config=FAST_FETCH)
    assert attempts["count"] == FAST_FETCH.max_retries


def test_fetch_records_outcomes_to_breaker():
    breaker = CircuitBreaker(threshold=10)

    def ok():
        return 1

    def bad():
        raise PermanentError("nope")

    fetch_with_retry(ok, breaker, fetch_config=FAST_FETCH)
    assert breaker.consecutive_failures == 0

    with pytest.raises(PermanentError):
        fetch_with_retry(bad, breaker, fetch_config=FAST_FETCH)
    assert breaker.consecutive_failures == 1

    fetch_with_retry(ok, breaker, fetch_config=FAST_FETCH)
    assert breaker.consecutive_failures == 0


def test_fetch_aborts_when_circuit_already_open():
    breaker = CircuitBreaker(threshold=2)
    breaker.record_failure()
    breaker.record_failure()

    def should_not_run():
        raise AssertionError("fetcher must short-circuit when breaker is open")

    with pytest.raises(CircuitOpenError):
        fetch_with_retry(should_not_run, breaker, fetch_config=FAST_FETCH)
