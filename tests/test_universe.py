"""Universe snapshot loader tests.

Run against the committed snapshot files. They do not hit the network; that
happens only in `_build.py` when explicitly refreshing the universe.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from graphsignal.data import universe as uni

VALID_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{0,5}$")


def test_sp500_snapshot_loads_with_expected_schema():
    df = uni.load_sp500()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "ticker", "name", "sector", "weight_pct", "source", "retrieved_at",
    }
    # Roughly 500 names, allow a small band for share-class duplicates / drift.
    assert 480 <= len(df) <= 520
    assert df["ticker"].is_unique
    assert df["ticker"].apply(lambda t: bool(VALID_TICKER_RE.match(t))).all()


def test_etfs_snapshot_loads_with_expected_schema():
    df = uni.load_etfs()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "ticker", "name", "category", "avg_dollar_volume", "rank",
        "source", "retrieved_at",
    }
    assert len(df) == 50
    assert df["ticker"].is_unique
    assert df["rank"].tolist() == list(range(1, 51))
    # Ranking is descending in volume.
    assert df["avg_dollar_volume"].is_monotonic_decreasing


def test_load_universe_default_returns_combined_tickers():
    sp = uni.load_universe("sp500")
    etfs = uni.load_universe("etfs")
    combined = uni.load_universe("all")
    assert combined == sp + etfs
    # Combined list should also be unique.
    assert len(set(combined)) == len(combined)


def test_load_universe_default_kind_is_all():
    assert uni.load_universe() == uni.load_universe("all")


def test_load_universe_rejects_unknown_kind():
    with pytest.raises(ValueError):
        uni.load_universe("crypto")  # type: ignore[arg-type]


def test_load_universe_as_of_not_implemented():
    with pytest.raises(NotImplementedError):
        uni.load_universe(as_of="2020-01-01")


def test_missing_snapshot_raises_clear_error(tmp_path, monkeypatch):
    """If a snapshot file is missing, the error should point at the build script."""
    fake = tmp_path / "nonexistent.csv"
    monkeypatch.setattr(uni, "SP500_FILE", fake)
    with pytest.raises(uni.UniverseSnapshotMissing) as excinfo:
        uni.load_sp500()
    assert "_build" in str(excinfo.value)


def test_etf_candidates_file_is_committed_and_parseable():
    """The candidate seed list must remain readable; the build script needs it."""
    df = pd.read_csv(uni.ETF_CANDIDATES_FILE)
    assert {"ticker", "name", "category"} <= set(df.columns)
    assert len(df) >= 60  # we ship ~80; allow trimming
    assert df["ticker"].apply(lambda t: bool(VALID_TICKER_RE.match(t))).all()
