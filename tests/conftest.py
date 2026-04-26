"""Shared Phase 0 test fixtures and data builders."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from graphsignal.config import EventThresholds, FetchConfig, GraphSignalConfig


@pytest.fixture
def fast_fetch_config() -> FetchConfig:
    return FetchConfig(
        max_retries=1,
        initial_backoff_seconds=0.001,
        max_backoff_seconds=0.005,
        concurrency=1,
        circuit_breaker_consecutive_failures=10,
    )


@pytest.fixture
def cfg(tmp_path, fast_fetch_config: FetchConfig) -> GraphSignalConfig:
    return GraphSignalConfig(
        data_dir=tmp_path / "data",
        universe_dir=tmp_path / "universe",
        fetch=fast_fetch_config,
        events=EventThresholds(
            drawdown_threshold_pct=0.20,
            drawdown_window_days=3,
            gap_threshold_pct=0.05,
            volume_spike_zscore=2.0,
            volume_spike_window_days=3,
        ),
    )


@pytest.fixture
def price_row() -> Callable[..., dict]:
    def _build(
        date: str,
        close: float,
        *,
        open_: float | None = None,
        volume: int = 1000,
    ) -> dict:
        px_open = close if open_ is None else open_
        return {
            "date": date,
            "open": px_open,
            "high": max(px_open, close),
            "low": min(px_open, close),
            "close": close,
            "volume": volume,
            "adj_close": close,
        }

    return _build


@pytest.fixture
def fake_yf_frame() -> Callable[..., pd.DataFrame]:
    def _build(
        ticker: str,
        start: str,
        end: str,
        *,
        base_close: float = 100.0,
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start, end=end, freq="B")
        n = len(dates)
        if n == 0:
            return pd.DataFrame()
        rows = {
            "Open": [base_close + i for i in range(n)],
            "High": [base_close + i + 1 for i in range(n)],
            "Low": [base_close + i - 1 for i in range(n)],
            "Close": [base_close + i + 0.5 for i in range(n)],
            "Adj Close": [base_close + i + 0.5 for i in range(n)],
            "Volume": [1_000_000 + i for i in range(n)],
        }
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="Date"))
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        return df

    return _build


@pytest.fixture
def slice_by_range() -> Callable[[pd.DataFrame, str, str], pd.DataFrame]:
    def _slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        mask = (frame.index >= pd.to_datetime(start)) & (frame.index < pd.to_datetime(end))
        return frame.loc[mask]

    return _slice


@pytest.fixture
def make_fake_download(
    slice_by_range: Callable[[pd.DataFrame, str, str], pd.DataFrame],
) -> Callable[[dict[str, pd.DataFrame]], Callable[..., pd.DataFrame]]:
    def _make(per_ticker: dict[str, pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        def _fake(tickers, start, end):
            ticker = tickers[0] if isinstance(tickers, list) else tickers
            if ticker not in per_ticker:
                return pd.DataFrame()
            return slice_by_range(per_ticker[ticker], start, end)

        return _fake

    return _make


@pytest.fixture
def fake_holdings_df() -> Callable[[list[tuple[str, float]]], pd.DataFrame]:
    def _build(symbols_and_pct: list[tuple[str, float]]) -> pd.DataFrame:
        return pd.DataFrame({
            "Symbol": [s for s, _ in symbols_and_pct],
            "Name": [f"{s} Inc" for s, _ in symbols_and_pct],
            "Holding Percent": [p for _, p in symbols_and_pct],
        }).set_index("Symbol")

    return _build
