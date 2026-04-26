"""Phase 0 local data-store workflow contracts."""
from __future__ import annotations

from datetime import date

import pandas as pd

from graphsignal.data import ingest as ingest_module
from graphsignal.data.events import load_events
from graphsignal.data.ingest import data_store_manifest, ingest_all
from graphsignal.data.prices import load_prices
from graphsignal.data.reference import (
    load_etf_constituents,
    load_etf_metadata,
    load_sectors,
)


def _eventful_yf_frame(ticker: str, *, base: float = 100.0) -> pd.DataFrame:
    """Build a yfinance-shaped frame that creates drawdown and gap events."""
    close = [base, base + 5, base, base * 0.80, base * 0.79]
    open_ = [base, base, base, base + 6, base * 0.79]
    rows = pd.DataFrame({
        "Open": open_,
        "High": [max(o, c) for o, c in zip(open_, close)],
        "Low": [min(o, c) for o, c in zip(open_, close)],
        "Close": close,
        "Adj Close": close,
        "Volume": [100, 110, 90, 100, 500],
    }, index=pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="D"), name="Date"))
    rows.columns = pd.MultiIndex.from_product([[ticker], rows.columns])
    return rows


def _install_fake_upstreams(
    monkeypatch,
    make_fake_download,
    fake_holdings_df,
    *,
    calls: dict[str, int] | None = None,
) -> None:
    price_frames = {
        "AAPL": _eventful_yf_frame("AAPL", base=100.0),
        "MSFT": _eventful_yf_frame("MSFT", base=120.0),
        "SPY": _eventful_yf_frame("SPY", base=400.0),
    }
    fake_info = {
        "AAPL": {"sector": "Technology", "industry": "Consumer Electronics", "quoteType": "EQUITY"},
        "MSFT": {"sector": "Technology", "industry": "Software", "quoteType": "EQUITY"},
        "SPY": {
            "quoteType": "ETF",
            "longName": "SPDR S&P 500 ETF Trust",
            "fundInceptionDate": 727660800,
        },
    }

    def fake_download(tickers, start, end):
        if calls is not None:
            calls["prices"] += 1
        return make_fake_download(price_frames)(tickers, start, end)

    def fake_info_for(ticker):
        if calls is not None:
            calls["info"] += 1
        return fake_info[ticker]

    def fake_holdings(etf):
        if calls is not None:
            calls["holdings"] += 1
        return fake_holdings_df([("AAPL", 0.07), ("MSFT", 0.06)])

    def fake_earnings(ticker):
        if calls is not None:
            calls["earnings"] += 1
        return pd.DataFrame({"Earnings Date": pd.to_datetime(["2024-01-25"])})

    monkeypatch.setattr(ingest_module.prices, "_yfinance_download", fake_download)
    monkeypatch.setattr(ingest_module.reference, "_yfinance_info", fake_info_for)
    monkeypatch.setattr(ingest_module.reference, "_yfinance_top_holdings", fake_holdings)
    monkeypatch.setattr(ingest_module.events, "_yfinance_earnings_dates", fake_earnings)


def test_ingest_all_builds_loadable_phase0_store(
    monkeypatch,
    cfg,
    make_fake_download,
    fake_holdings_df,
):
    _install_fake_upstreams(monkeypatch, make_fake_download, fake_holdings_df)

    ingest_all(
        ["AAPL", "MSFT", "SPY"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 5),
        config=cfg,
    )

    assert set(load_prices(["AAPL", "MSFT", "SPY"], config=cfg).df["ticker"]) == {
        "AAPL",
        "MSFT",
        "SPY",
    }
    assert set(load_sectors(config=cfg)["ticker"]) == {"AAPL", "MSFT", "SPY"}
    assert load_etf_metadata(config=cfg)["ticker"].tolist() == ["SPY"]
    assert load_etf_constituents("SPY", config=cfg) == ["AAPL", "MSFT"]
    assert load_events("earnings", config=cfg)["ticker"].unique().tolist() == [
        "AAPL",
        "MSFT",
        "SPY",
    ]
    assert {"drawdowns", "gaps"} <= set(load_events("all", config=cfg)["event_type"])


def test_ingest_all_rerun_uses_checkpoints_and_preserves_manifest(
    monkeypatch,
    cfg,
    make_fake_download,
    fake_holdings_df,
):
    calls = {"prices": 0, "info": 0, "holdings": 0, "earnings": 0}
    _install_fake_upstreams(
        monkeypatch,
        make_fake_download,
        fake_holdings_df,
        calls=calls,
    )

    kwargs = {
        "tickers": ["AAPL", "MSFT", "SPY"],
        "start_date": date(2024, 1, 1),
        "end_date": date(2024, 1, 5),
        "config": cfg,
    }
    ingest_all(**kwargs)
    first_manifest = data_store_manifest(cfg.data_dir)
    calls_after_first_run = calls.copy()
    ingest_all(**kwargs)
    second_manifest = data_store_manifest(cfg.data_dir)

    pd.testing.assert_frame_equal(first_manifest, second_manifest)
    assert calls == calls_after_first_run


def test_skip_earnings_still_derives_events_from_local_prices(
    monkeypatch,
    cfg,
    make_fake_download,
    fake_holdings_df,
):
    _install_fake_upstreams(monkeypatch, make_fake_download, fake_holdings_df)

    summary = ingest_all(
        ["AAPL", "MSFT", "SPY"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 5),
        skip_earnings=True,
        config=cfg,
    )

    assert summary.events.earnings_updated == []
    assert load_events("earnings", config=cfg).empty
    assert {"drawdowns", "gaps"} <= set(load_events("all", config=cfg)["event_type"])
