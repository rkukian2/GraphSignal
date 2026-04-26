"""Price ingestion + loader tests. No network access; yfinance is mocked."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from graphsignal.config import FetchConfig, GraphSignalConfig
from graphsignal.data import prices
from graphsignal.data._fetch import PermanentError
from graphsignal.data.prices import (
    PRICE_COLUMNS,
    _normalize_yf_frame,
    _write_parquet_atomic,
    ingest_prices,
    load_prices,
)


# --- helpers ----------------------------------------------------------------

def _fake_yf_frame(
    ticker: str,
    start: str,
    end: str,
    *,
    base_close: float = 100.0,
) -> pd.DataFrame:
    """Build a yfinance-style multi-index DataFrame for one ticker."""
    dates = pd.date_range(start=start, end=end, freq="B")
    n = len(dates)
    if n == 0:
        return pd.DataFrame()
    rows = {
        "Open":      [base_close + i for i in range(n)],
        "High":      [base_close + i + 1 for i in range(n)],
        "Low":       [base_close + i - 1 for i in range(n)],
        "Close":     [base_close + i + 0.5 for i in range(n)],
        "Adj Close": [base_close + i + 0.5 for i in range(n)],
        "Volume":    [1_000_000 + i for i in range(n)],
    }
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="Date"))
    df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
    return df


def _slice_by_range(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (frame.index >= pd.to_datetime(start)) & (
        frame.index < pd.to_datetime(end)
    )
    return frame.loc[mask]


def _make_fake_download(per_ticker: dict[str, pd.DataFrame]):
    def _fake(tickers, start, end):
        ticker = tickers[0] if isinstance(tickers, list) else tickers
        if ticker not in per_ticker:
            return pd.DataFrame()
        return _slice_by_range(per_ticker[ticker], start, end)
    return _fake


def _fast_fetch_config() -> FetchConfig:
    return FetchConfig(
        max_retries=1,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.005,
        concurrency=1,
        circuit_breaker_consecutive_failures=10,
    )


@pytest.fixture
def cfg(tmp_path) -> GraphSignalConfig:
    return GraphSignalConfig(
        data_dir=tmp_path / "data",
        universe_dir=tmp_path / "universe",
        fetch=_fast_fetch_config(),
    )


# --- normalizer -------------------------------------------------------------

def test_normalize_handles_multiindex_ticker_first():
    raw = _fake_yf_frame("AAPL", "2024-01-01", "2024-01-10")
    df = _normalize_yf_frame(raw, "AAPL")
    assert list(df.columns) == PRICE_COLUMNS
    assert df["close"].notna().all()
    assert df["volume"].dtype.kind in "iu"
    assert (df["date"].diff().dropna() > pd.Timedelta(0)).all()


def test_normalize_rejects_empty_frame():
    with pytest.raises(PermanentError):
        _normalize_yf_frame(pd.DataFrame(), "AAPL")


def test_normalize_rejects_missing_column():
    raw = _fake_yf_frame("AAPL", "2024-01-01", "2024-01-05")
    raw = raw.drop(columns=("AAPL", "Volume"))
    with pytest.raises(PermanentError):
        _normalize_yf_frame(raw, "AAPL")


# --- storage round trip -----------------------------------------------------

def test_write_then_read_round_trip(cfg):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "open": [100.0, 101.0],
        "high": [101.0, 102.0],
        "low":  [99.0, 100.0],
        "close": [100.5, 101.5],
        "volume": [1000, 1100],
        "adj_close": [100.5, 101.5],
    })
    path = cfg.data_dir / "prices" / "AAPL.parquet"
    _write_parquet_atomic(path, df)
    assert path.exists()

    result = load_prices("AAPL", config=cfg, adjusted=False)
    assert result.missing == []
    assert len(result.df) == 2
    assert list(result.df["close"]) == [100.5, 101.5]


# --- loader -----------------------------------------------------------------

def test_load_prices_missing_ticker(cfg):
    result = load_prices(["NONEXISTENT"], config=cfg)
    assert result.df.empty
    assert result.missing == ["NONEXISTENT"]


def test_load_prices_filters_date_range(cfg):
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-02"]),
        "open": [100.0, 110.0, 120.0],
        "high": [101.0, 111.0, 121.0],
        "low":  [99.0, 109.0, 119.0],
        "close": [100.5, 110.5, 120.5],
        "volume": [1000, 2000, 3000],
        "adj_close": [100.5, 110.5, 120.5],
    })
    _write_parquet_atomic(cfg.data_dir / "prices" / "AAPL.parquet", df)

    result = load_prices(
        "AAPL",
        start="2024-02-01",
        end="2024-02-15",
        config=cfg,
        adjusted=False,
    )
    assert len(result.df) == 1
    assert result.df["date"].iloc[0] == pd.Timestamp("2024-02-02")


def test_load_prices_adjusted_scales_ohlc(cfg):
    """When adj_close differs from close (e.g. a 2:1 split), adjusted=True
    scales OHLC by the same ratio."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "open": [200.0],
        "high": [205.0],
        "low":  [195.0],
        "close": [200.0],
        "volume": [1000],
        "adj_close": [100.0],
    })
    _write_parquet_atomic(cfg.data_dir / "prices" / "X.parquet", df)

    r_unadj = load_prices("X", config=cfg, adjusted=False)
    assert r_unadj.df["close"].iloc[0] == 200.0
    assert r_unadj.df["high"].iloc[0] == 205.0

    r_adj = load_prices("X", config=cfg, adjusted=True)
    # ratio = 100 / 200 = 0.5
    assert r_adj.df["close"].iloc[0] == 100.0
    assert r_adj.df["open"].iloc[0] == 100.0
    assert r_adj.df["high"].iloc[0] == 102.5
    assert r_adj.df["low"].iloc[0] == 97.5


# --- ingestion --------------------------------------------------------------

def test_ingest_prices_writes_new_file(monkeypatch, cfg):
    fake = {"AAPL": _fake_yf_frame("AAPL", "2024-01-01", "2024-01-10")}
    monkeypatch.setattr(prices, "_yfinance_download", _make_fake_download(fake))

    s = ingest_prices(
        ["AAPL"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        config=cfg,
    )
    assert "AAPL" in s.updated
    assert s.skipped == []
    assert (cfg.data_dir / "prices" / "AAPL.parquet").exists()


def test_ingest_prices_idempotent(monkeypatch, cfg):
    """Two runs against the same data leave the Parquet byte-equivalent."""
    fake = {"AAPL": _fake_yf_frame("AAPL", "2024-01-01", "2024-01-10")}
    monkeypatch.setattr(prices, "_yfinance_download", _make_fake_download(fake))

    ingest_prices(
        ["AAPL"], start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10), config=cfg,
    )
    parquet_path = cfg.data_dir / "prices" / "AAPL.parquet"
    bytes_first = parquet_path.read_bytes()

    s = ingest_prices(
        ["AAPL"], start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10), config=cfg,
    )
    bytes_second = parquet_path.read_bytes()

    assert "AAPL" in s.already_current
    assert s.updated == []
    assert bytes_first == bytes_second


def test_ingest_prices_appends_new_data(monkeypatch, cfg):
    """A second run with a later end_date appends new rows for the same ticker."""
    full = _fake_yf_frame("AAPL", "2024-01-01", "2024-01-20")

    def fake(tickers, start, end):
        return _slice_by_range(full, start, end)

    monkeypatch.setattr(prices, "_yfinance_download", fake)

    ingest_prices(
        ["AAPL"], start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10), config=cfg,
    )
    initial_len = len(load_prices("AAPL", config=cfg, adjusted=False).df)

    ingest_prices(
        ["AAPL"], start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 20), config=cfg,
    )
    final_df = load_prices("AAPL", config=cfg, adjusted=False).df
    assert len(final_df) > initial_len
    assert final_df["date"].is_monotonic_increasing
    assert final_df["date"].is_unique


def test_ingest_prices_skips_missing_ticker(monkeypatch, cfg):
    fake = {"AAPL": _fake_yf_frame("AAPL", "2024-01-01", "2024-01-10")}
    monkeypatch.setattr(prices, "_yfinance_download", _make_fake_download(fake))

    s = ingest_prices(
        ["AAPL", "FAKE"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        config=cfg,
    )
    assert "AAPL" in s.updated
    assert any(t == "FAKE" for t, _ in s.skipped)
    assert s.circuit_open is False


def test_ingest_prices_circuit_breaker_halts_run(monkeypatch, cfg):
    """Many consecutive failures trip the breaker mid-run."""
    monkeypatch.setattr(
        prices, "_yfinance_download",
        lambda tickers, start, end: pd.DataFrame(),
    )
    cfg2 = cfg.model_copy(update={
        "fetch": FetchConfig(
            max_retries=1,
            initial_backoff_seconds=0.001,
            max_backoff_seconds=0.005,
            concurrency=1,
            circuit_breaker_consecutive_failures=3,
        ),
    })

    s = ingest_prices(
        ["A", "B", "C", "D", "E", "F"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        config=cfg2,
    )
    assert s.circuit_open is True
    # First 3 failures recorded as skipped; loop halts before D, E, F are tried.
    assert len(s.skipped) == 3
    assert {t for t, _ in s.skipped} == {"A", "B", "C"}
