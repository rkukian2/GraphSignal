"""Event metadata ingestion and loaders.

Phase 0 stores four event types under `<data_dir>/events/`:

- earnings: best-effort yfinance earnings dates.
- drawdowns: derived from adjusted close.
- gaps: derived from raw open versus prior raw close.
- volume_spikes: derived from raw volume.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import pandas as pd
import yfinance as yf

from graphsignal.config import GraphSignalConfig, get_config
from graphsignal.data._fetch import (
    CircuitBreaker,
    CircuitOpenError,
    PermanentError,
    TransientError,
    fetch_with_retry,
)
from graphsignal.data._storage import (
    merge_snapshot,
    missing_from_snapshot,
    read_parquet_or_empty,
    unique_ordered,
    write_dataframe_atomic,
)
from graphsignal.data.prices import load_prices
from graphsignal.data.universe import load_universe

log = logging.getLogger(__name__)

EventType = Literal["earnings", "drawdowns", "gaps", "volume_spikes", "all"]
DateLike = Union[str, pd.Timestamp]

COMMON_COLUMNS = ["event_type", "ticker", "date", "source", "created_at"]
EARNINGS_COLUMNS = [*COMMON_COLUMNS, "event_time"]
DRAWDOWN_COLUMNS = [
    *COMMON_COLUMNS,
    "window_days",
    "threshold_pct",
    "drawdown_pct",
    "peak_date",
]
GAP_COLUMNS = [
    *COMMON_COLUMNS,
    "threshold_pct",
    "gap_pct",
    "prior_close",
    "open",
]
VOLUME_SPIKE_COLUMNS = [
    *COMMON_COLUMNS,
    "window_days",
    "zscore_threshold",
    "volume_zscore",
    "volume",
    "trailing_mean",
    "trailing_std",
]

EVENT_COLUMNS: dict[str, list[str]] = {
    "earnings": EARNINGS_COLUMNS,
    "drawdowns": DRAWDOWN_COLUMNS,
    "gaps": GAP_COLUMNS,
    "volume_spikes": VOLUME_SPIKE_COLUMNS,
}


@dataclass
class EventIngestionSummary:
    earnings_updated: list[str] = field(default_factory=list)
    earnings_already_current: list[str] = field(default_factory=list)
    earnings_skipped: list[tuple[str, str]] = field(default_factory=list)
    earnings_circuit_open: bool = False
    derived_updated: list[str] = field(default_factory=list)
    derived_missing_prices: list[str] = field(default_factory=list)


def _events_dir(cfg: GraphSignalConfig) -> Path:
    return cfg.data_dir / "events"


def _event_path(cfg: GraphSignalConfig, event_type: str) -> Path:
    return _events_dir(cfg) / f"{event_type}.parquet"


def _created_at() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _empty_events(event_type: str) -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS[event_type])


def _read_events(cfg: GraphSignalConfig, event_type: str) -> pd.DataFrame:
    return read_parquet_or_empty(_event_path(cfg, event_type), EVENT_COLUMNS[event_type])


def _write_events_atomic(cfg: GraphSignalConfig, event_type: str, df: pd.DataFrame) -> None:
    write_dataframe_atomic(
        _event_path(cfg, event_type),
        df,
        EVENT_COLUMNS[event_type],
        date_columns=["date", "peak_date"],
    )


def _canonical_for_compare(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    out = df[EVENT_COLUMNS[event_type]].copy() if not df.empty else _empty_events(event_type)
    for col in ("date", "peak_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out.sort_values([c for c in ("ticker", "date") if c in out.columns]).reset_index(drop=True)


def _concat_or_empty(frames: Sequence[pd.DataFrame], event_type: str) -> pd.DataFrame:
    non_empty = [df for df in frames if not df.empty]
    if not non_empty:
        return _empty_events(event_type)
    return pd.concat(non_empty, ignore_index=True)


def _yfinance_earnings_dates(ticker: str) -> pd.DataFrame:
    """Fetch yfinance earnings dates for one ticker. Test seam."""
    df = yf.Ticker(ticker).get_earnings_dates(limit=1000)
    if df is None:
        return pd.DataFrame()
    return df


def _normalize_earnings_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise PermanentError(f"{ticker}: no earnings dates returned")

    df = raw.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    date_col = next(
        (
            c for c in df.columns
            if str(c).lower() in {"earnings date", "earnings_date", "date", "index"}
        ),
        df.columns[0] if len(df.columns) else None,
    )
    if date_col is None:
        raise PermanentError(f"{ticker}: earnings dates missing date column")

    event_time_col = next((c for c in df.columns if str(c).lower() == "event time"), None)
    dates = (
        pd.to_datetime(df[date_col], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    out = pd.DataFrame({
        "event_type": "earnings",
        "ticker": ticker,
        "date": dates,
        "source": "yfinance earnings dates",
        "created_at": _created_at(),
        "event_time": df[event_time_col] if event_time_col else pd.NA,
    })
    out = out.dropna(subset=["date"])
    if out.empty:
        raise PermanentError(f"{ticker}: earnings dates malformed")
    return out[EARNINGS_COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def ingest_earnings(
    tickers: Optional[Sequence[str]] = None,
    *,
    force: bool = False,
    config: Optional[GraphSignalConfig] = None,
) -> tuple[list[str], list[str], list[tuple[str, str]], bool]:
    """Fetch best-effort earnings dates and persist them.

    Returns (updated, already_current, skipped, circuit_open). Existing rows are
    checkpoints unless `force=True`.
    """
    cfg = config or get_config()
    if tickers is None:
        tickers = load_universe("all")
    tickers = unique_ordered(tickers)
    existing = _read_events(cfg, "earnings")
    fetch_tickers = tickers if force else missing_from_snapshot(tickers, existing)
    already_current = [] if force else [t for t in tickers if t not in fetch_tickers]

    if not fetch_tickers:
        return [], already_current, [], False

    breaker = CircuitBreaker(cfg.fetch.circuit_breaker_consecutive_failures)
    rows: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []
    circuit_open = False

    for ticker in fetch_tickers:
        try:
            breaker.check()
        except CircuitOpenError:
            circuit_open = True
            log.error("circuit open; halting earnings ingestion")
            break

        def _do() -> pd.DataFrame:
            try:
                return _yfinance_earnings_dates(ticker)
            except Exception as e:
                raise TransientError(f"{ticker}: yfinance earnings dates: {e}") from e

        try:
            raw = fetch_with_retry(_do, breaker, fetch_config=cfg.fetch, label=f"earnings-{ticker}")
            normalized = _normalize_earnings_frame(raw, ticker)
        except (PermanentError, TransientError) as e:
            skipped.append((ticker, str(e)))
            log.warning("skipping earnings for %s: %s", ticker, e)
            continue
        except CircuitOpenError:
            circuit_open = True
            break
        rows.append(normalized)

    updated = sorted({df["ticker"].iloc[0] for df in rows})
    if rows:
        new_df = pd.concat(rows, ignore_index=True)
        out = merge_snapshot(
            existing,
            new_df,
            EARNINGS_COLUMNS,
            key_columns=["ticker", "date"],
            refreshed_values=tickers if force else None,
            sort_by=["ticker", "date"],
        )
        _write_events_atomic(cfg, "earnings", out)

    return updated, already_current, skipped, circuit_open


def _common(event_type: str, ticker: str, n: int) -> dict:
    return {
        "event_type": [event_type] * n,
        "ticker": [ticker] * n,
        "source": ["derived from local prices"] * n,
        "created_at": [_created_at()] * n,
    }


def _compute_drawdowns(df: pd.DataFrame, ticker: str, cfg: GraphSignalConfig) -> pd.DataFrame:
    thresholds = cfg.events
    work = df.sort_values("date").reset_index(drop=True).copy()
    rolling_peak = work["adj_close"].rolling(
        window=thresholds.drawdown_window_days,
        min_periods=1,
    ).max()
    peak_idx = work["adj_close"].rolling(
        window=thresholds.drawdown_window_days,
        min_periods=1,
    ).apply(lambda x: float(x.idxmax()), raw=False)
    work["drawdown_pct"] = work["adj_close"] / rolling_peak - 1.0
    mask = work["drawdown_pct"] <= -thresholds.drawdown_threshold_pct
    if not mask.any():
        return _empty_events("drawdowns")
    out = pd.DataFrame(_common("drawdowns", ticker, int(mask.sum())))
    selected = work.loc[mask]
    out["date"] = selected["date"].to_list()
    out["window_days"] = thresholds.drawdown_window_days
    out["threshold_pct"] = thresholds.drawdown_threshold_pct
    out["drawdown_pct"] = selected["drawdown_pct"].to_list()
    out["peak_date"] = [work.loc[int(i), "date"] for i in peak_idx.loc[mask]]
    return out[DRAWDOWN_COLUMNS]


def _compute_gaps(df: pd.DataFrame, ticker: str, cfg: GraphSignalConfig) -> pd.DataFrame:
    thresholds = cfg.events
    work = df.sort_values("date").reset_index(drop=True).copy()
    work["prior_close"] = work["close"].shift(1)
    work["gap_pct"] = work["open"] / work["prior_close"] - 1.0
    mask = work["prior_close"].notna() & (work["gap_pct"].abs() >= thresholds.gap_threshold_pct)
    if not mask.any():
        return _empty_events("gaps")
    selected = work.loc[mask]
    out = pd.DataFrame(_common("gaps", ticker, len(selected)))
    out["date"] = selected["date"].to_list()
    out["threshold_pct"] = thresholds.gap_threshold_pct
    out["gap_pct"] = selected["gap_pct"].to_list()
    out["prior_close"] = selected["prior_close"].to_list()
    out["open"] = selected["open"].to_list()
    return out[GAP_COLUMNS]


def _compute_volume_spikes(df: pd.DataFrame, ticker: str, cfg: GraphSignalConfig) -> pd.DataFrame:
    thresholds = cfg.events
    work = df.sort_values("date").reset_index(drop=True).copy()
    trailing = work["volume"].shift(1).rolling(
        window=thresholds.volume_spike_window_days,
        min_periods=thresholds.volume_spike_window_days,
    )
    work["trailing_mean"] = trailing.mean()
    work["trailing_std"] = trailing.std()
    work["volume_zscore"] = (work["volume"] - work["trailing_mean"]) / work["trailing_std"]
    mask = (
        work["trailing_std"].notna()
        & (work["trailing_std"] > 0)
        & (work["volume_zscore"] >= thresholds.volume_spike_zscore)
    )
    if not mask.any():
        return _empty_events("volume_spikes")
    selected = work.loc[mask]
    out = pd.DataFrame(_common("volume_spikes", ticker, len(selected)))
    out["date"] = selected["date"].to_list()
    out["window_days"] = thresholds.volume_spike_window_days
    out["zscore_threshold"] = thresholds.volume_spike_zscore
    out["volume_zscore"] = selected["volume_zscore"].to_list()
    out["volume"] = selected["volume"].to_list()
    out["trailing_mean"] = selected["trailing_mean"].to_list()
    out["trailing_std"] = selected["trailing_std"].to_list()
    return out[VOLUME_SPIKE_COLUMNS]


def ingest_derived_events(
    tickers: Optional[Sequence[str]] = None,
    *,
    force: bool = False,
    config: Optional[GraphSignalConfig] = None,
) -> tuple[list[str], list[str]]:
    """Compute drawdowns, gaps, and volume spikes from local price files."""
    cfg = config or get_config()
    if tickers is None:
        tickers = load_universe("all")
    tickers = unique_ordered(tickers)

    loaded = load_prices(tickers, adjusted=False, config=cfg)
    if loaded.df.empty:
        for event_type in ("drawdowns", "gaps", "volume_spikes"):
            existing = _read_events(cfg, event_type)
            if force or existing.empty:
                _write_events_atomic(cfg, event_type, existing)
        return ["drawdowns", "gaps", "volume_spikes"], loaded.missing

    frames: dict[str, list[pd.DataFrame]] = {
        "drawdowns": [],
        "gaps": [],
        "volume_spikes": [],
    }
    for ticker, df in loaded.df.groupby("ticker", sort=True):
        frames["drawdowns"].append(_compute_drawdowns(df, ticker, cfg))
        frames["gaps"].append(_compute_gaps(df, ticker, cfg))
        frames["volume_spikes"].append(_compute_volume_spikes(df, ticker, cfg))

    updated: list[str] = []
    for event_type, event_frames in frames.items():
        new_df = _concat_or_empty(event_frames, event_type)
        existing = _read_events(cfg, event_type)
        out = merge_snapshot(
            existing,
            new_df,
            EVENT_COLUMNS[event_type],
            key_columns=["ticker", "date"],
            refreshed_values=tickers,
            sort_by=["ticker", "date"],
        )
        if not _canonical_for_compare(existing, event_type).equals(
            _canonical_for_compare(out, event_type)
        ):
            _write_events_atomic(cfg, event_type, out)
        updated.append(event_type)

    return updated, loaded.missing


def ingest_events(
    tickers: Optional[Sequence[str]] = None,
    *,
    force: bool = False,
    config: Optional[GraphSignalConfig] = None,
) -> EventIngestionSummary:
    cfg = config or get_config()
    summary = EventIngestionSummary()
    (
        summary.earnings_updated,
        summary.earnings_already_current,
        summary.earnings_skipped,
        summary.earnings_circuit_open,
    ) = ingest_earnings(tickers=tickers, force=force, config=cfg)
    (
        summary.derived_updated,
        summary.derived_missing_prices,
    ) = ingest_derived_events(tickers=tickers, force=force, config=cfg)
    return summary


def load_events(
    event_type: EventType,
    tickers: Optional[Union[Sequence[str], str]] = None,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    *,
    config: Optional[GraphSignalConfig] = None,
) -> pd.DataFrame:
    """Load events, optionally filtered by ticker and date range."""
    cfg = config or get_config()
    if event_type == "all":
        frames = [_read_events(cfg, t) for t in EVENT_COLUMNS]
        non_empty = [df for df in frames if not df.empty]
        if non_empty:
            df = pd.concat(non_empty, ignore_index=True, sort=False)
        else:
            df = pd.DataFrame(columns=COMMON_COLUMNS)
    elif event_type in EVENT_COLUMNS:
        df = _read_events(cfg, event_type)
    else:
        raise ValueError(f"unknown event_type: {event_type!r}")

    if df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if tickers is not None:
        wanted = {tickers} if isinstance(tickers, str) else set(tickers)
        out = out[out["ticker"].isin(wanted)]
    if start is not None:
        out = out[out["date"] >= pd.to_datetime(start)]
    if end is not None:
        out = out[out["date"] <= pd.to_datetime(end)]
    return out.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = ingest_events()
    log.info(
        "earnings: %d updated, %d already-current, %d skipped, circuit_open=%s",
        len(summary.earnings_updated),
        len(summary.earnings_already_current),
        len(summary.earnings_skipped),
        summary.earnings_circuit_open,
    )
    log.info(
        "derived: %s updated, %d tickers missing prices",
        ", ".join(summary.derived_updated),
        len(summary.derived_missing_prices),
    )


if __name__ == "__main__":
    main()
