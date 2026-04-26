"""Focused Phase 0 component contracts."""
from __future__ import annotations

import pandas as pd
import pytest

from graphsignal.data.events import (
    _compute_drawdowns,
    _compute_gaps,
    _compute_volume_spikes,
    _normalize_earnings_frame,
)
from graphsignal.data.prices import _normalize_yf_frame, _write_parquet_atomic, load_prices
from graphsignal.data.reference import _normalize_holdings


def test_price_payload_normalizes_to_local_schema(fake_yf_frame):
    df = _normalize_yf_frame(fake_yf_frame("AAPL", "2024-01-01", "2024-01-03"), "AAPL")

    assert df.to_dict("records") == [
        {
            "date": pd.Timestamp("2024-01-01"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
            "adj_close": 100.5,
        },
        {
            "date": pd.Timestamp("2024-01-02"),
            "open": 101.0,
            "high": 102.0,
            "low": 100.0,
            "close": 101.5,
            "volume": 1_000_001,
            "adj_close": 101.5,
        },
        {
            "date": pd.Timestamp("2024-01-03"),
            "open": 102.0,
            "high": 103.0,
            "low": 101.0,
            "close": 102.5,
            "volume": 1_000_002,
            "adj_close": 102.5,
        },
    ]


def test_adjusted_price_loader_scales_ohlc(cfg):
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "open": [200.0],
        "high": [210.0],
        "low": [190.0],
        "close": [200.0],
        "volume": [1000],
        "adj_close": [100.0],
    })
    _write_parquet_atomic(cfg.data_dir / "prices" / "SPLT.parquet", raw)

    loaded = load_prices("SPLT", adjusted=True, config=cfg).df.iloc[0]

    assert loaded["open"] == pytest.approx(100.0)
    assert loaded["high"] == pytest.approx(105.0)
    assert loaded["low"] == pytest.approx(95.0)
    assert loaded["close"] == pytest.approx(100.0)


def test_reference_payloads_normalize_holdings_and_earnings(fake_holdings_df):
    holdings = _normalize_holdings(fake_holdings_df([("AAPL", 0.07), ("MSFT", 0.05)]), "SPY")
    earnings = _normalize_earnings_frame(
        pd.DataFrame({
            "Earnings Date": pd.to_datetime(["2024-01-25"]),
            "Event Time": ["amc"],
        }),
        "AAPL",
    )

    assert holdings[["etf", "ticker"]].to_dict("records") == [
        {"etf": "SPY", "ticker": "AAPL"},
        {"etf": "SPY", "ticker": "MSFT"},
    ]
    assert holdings["weight_pct"].tolist() == pytest.approx([7.0, 5.0])
    assert earnings[["event_type", "ticker", "date", "event_time"]].to_dict("records") == [
        {
            "event_type": "earnings",
            "ticker": "AAPL",
            "date": pd.Timestamp("2024-01-25"),
            "event_time": "amc",
        }
    ]


def test_derived_event_calculations_emit_expected_signals(cfg, price_row):
    prices = pd.DataFrame([
        price_row("2024-01-01", 100, volume=100),
        price_row("2024-01-02", 105, open_=100, volume=110),
        price_row("2024-01-03", 100, volume=90),
        price_row("2024-01-04", 80, open_=106, volume=100),
        price_row("2024-01-05", 79, volume=500),
    ])
    prices["date"] = pd.to_datetime(prices["date"])

    drawdowns = _compute_drawdowns(prices, "AAA", cfg)
    gaps = _compute_gaps(prices, "AAA", cfg)
    spikes = _compute_volume_spikes(prices, "AAA", cfg)

    assert drawdowns["date"].tolist() == [
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-05"),
    ]
    assert drawdowns["drawdown_pct"].tolist() == pytest.approx([-25 / 105, -0.21])
    assert drawdowns["peak_date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert gaps[["date", "gap_pct", "prior_close", "open"]].to_dict("records") == [
        {
            "date": pd.Timestamp("2024-01-04"),
            "gap_pct": pytest.approx(0.06),
            "prior_close": 100.0,
            "open": 106.0,
        }
    ]
    assert spikes[["date", "volume"]].to_dict("records") == [
        {"date": pd.Timestamp("2024-01-05"), "volume": 500}
    ]
