"""Representative Phase 0 failure behavior."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from graphsignal.config import FetchConfig
from graphsignal.data import events, prices, reference
from graphsignal.data._fetch import PermanentError
from graphsignal.data.events import ingest_earnings, load_events
from graphsignal.data.prices import ingest_prices
from graphsignal.data.reference import (
    ingest_etf_constituents,
    load_etf_holdings,
    load_etf_metadata,
    load_sectors,
)


def test_price_ingestion_skips_empty_upstream_payload(monkeypatch, cfg):
    monkeypatch.setattr(prices, "_yfinance_download", lambda tickers, start, end: pd.DataFrame())

    summary = ingest_prices(
        ["MISSING"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 5),
        config=cfg,
    )

    assert summary.updated == []
    assert [ticker for ticker, _ in summary.skipped] == ["MISSING"]
    assert summary.circuit_open is False


def test_bad_event_and_reference_payloads_are_skipped(monkeypatch, cfg):
    monkeypatch.setattr(
        events,
        "_yfinance_earnings_dates",
        lambda ticker: pd.DataFrame({"Earnings Date": ["not-a-date"]}),
    )
    monkeypatch.setattr(
        reference,
        "_yfinance_top_holdings",
        lambda etf: (_ for _ in ()).throw(PermanentError(f"{etf}: no holdings")),
    )

    earnings_summary = ingest_earnings(["BAD"], config=cfg)
    holdings_summary = ingest_etf_constituents(["GLD"], config=cfg)

    assert earnings_summary[0] == []
    assert [ticker for ticker, _ in earnings_summary[2]] == ["BAD"]
    assert load_events("earnings", config=cfg).empty
    assert holdings_summary[0] == []
    assert [ticker for ticker, _ in holdings_summary[1]] == ["GLD"]


def test_circuit_breaker_halts_repeated_ingestion_failures(monkeypatch, cfg):
    monkeypatch.setattr(prices, "_yfinance_download", lambda tickers, start, end: pd.DataFrame())
    cfg = cfg.model_copy(update={
        "fetch": FetchConfig(
            max_retries=1,
            initial_backoff_seconds=0.001,
            max_backoff_seconds=0.005,
            concurrency=1,
            circuit_breaker_consecutive_failures=2,
        )
    })

    summary = ingest_prices(
        ["A", "B", "C"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 5),
        config=cfg,
    )

    assert summary.circuit_open is True
    assert [ticker for ticker, _ in summary.skipped] == ["A", "B"]


@pytest.mark.parametrize(
    ("loader", "args"),
    [(load_sectors, ()), (load_etf_metadata, ()), (load_etf_holdings, ("SPY",))],
)
def test_reference_loaders_raise_clear_error_when_snapshot_is_missing(cfg, loader, args):
    with pytest.raises(FileNotFoundError, match="Run: python -m graphsignal.data.reference"):
        loader(*args, config=cfg)
