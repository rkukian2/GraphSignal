"""Price ingestion and loaders for GraphSignal.

Stores daily OHLCV + adjusted close per ticker as Parquet files, one per
ticker, under `<data_dir>/prices/`.

- `ingest_prices(...)` fetches from yfinance through the smart fetcher with
  per-ticker idempotency: a re-run only fetches data dated after the last row
  on disk. Permanent failures (delisted tickers, schema breakage) are logged
  and skipped; the rest of the run continues. Too many consecutive failures
  trip the circuit breaker and halt the run cleanly.
- `load_prices(...)` reads back the Parquet files for downstream code, with
  optional split/dividend adjustment of OHLC.
- `LoadedPrices` is the typed return: a long-form DataFrame plus a list of
  tickers that were requested but absent from disk (or empty in the range).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

from graphsignal.config import GraphSignalConfig, get_config
from graphsignal.data._fetch import (
    CircuitBreaker,
    CircuitOpenError,
    PermanentError,
    TransientError,
    fetch_with_retry,
)
from graphsignal.data._storage import unique_ordered
from graphsignal.data.universe import load_universe

log = logging.getLogger(__name__)

DEFAULT_START_DATE = date(2000, 1, 1)
PRICE_COLUMNS: list[str] = [
    "date", "open", "high", "low", "close", "volume", "adj_close",
]
PRICE_SCHEMA = pa.schema([
    pa.field("date", pa.timestamp("ns"), nullable=False),
    pa.field("open", pa.float64(), nullable=True),
    pa.field("high", pa.float64(), nullable=True),
    pa.field("low", pa.float64(), nullable=True),
    pa.field("close", pa.float64(), nullable=True),
    pa.field("volume", pa.int64(), nullable=True),
    pa.field("adj_close", pa.float64(), nullable=True),
])

DateLike = Union[str, date, pd.Timestamp]


class LoadedPrices(NamedTuple):
    df: pd.DataFrame
    missing: list[str]


@dataclass
class IngestionSummary:
    updated: list[str] = field(default_factory=list)
    already_current: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    circuit_open: bool = False

    def total(self) -> int:
        return len(self.updated) + len(self.already_current) + len(self.skipped)


# --- paths ------------------------------------------------------------------

def _prices_dir(cfg: GraphSignalConfig) -> Path:
    return cfg.data_dir / "prices"


def _parquet_path(cfg: GraphSignalConfig, ticker: str) -> Path:
    return _prices_dir(cfg) / f"{ticker}.parquet"


# --- storage ----------------------------------------------------------------

def _read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _existing_last_date(path: Path) -> Optional[date]:
    if not path.exists():
        return None
    df = pq.read_table(path, columns=["date"]).to_pandas()
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max().date()


def _write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    """Write to a sibling .tmp file then rename. Avoids leaving a half-written
    Parquet on disk if the process is killed mid-write.
    """
    if df.empty:
        return
    out = df[PRICE_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["volume"] = out["volume"].astype("Int64").astype("int64")
    table = pa.Table.from_pandas(out, schema=PRICE_SCHEMA, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


# --- normalizer -------------------------------------------------------------

_YF_RENAME = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Adj Close": "adj_close",
}


def _normalize_yf_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert a yfinance result for one ticker into our schema.

    Accepts either flat columns (single-ticker call) or multi-index columns
    (batch call with group_by="ticker"). Empty results and missing columns
    raise PermanentError; those will not heal on retry.
    """
    if raw is None or raw.empty:
        raise PermanentError(f"yfinance returned no data for {ticker}")
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        levels0 = df.columns.get_level_values(0)
        levels_last = df.columns.get_level_values(-1)
        if ticker in levels0:
            df = df[ticker]
        elif ticker in levels_last:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            raise PermanentError(
                f"{ticker} not present in yfinance result columns"
            )
    df = df.reset_index()
    missing_cols = [c for c in _YF_RENAME if c not in df.columns]
    if missing_cols:
        raise PermanentError(
            f"{ticker}: yfinance result missing columns {missing_cols}"
        )
    df = df.rename(columns=_YF_RENAME)[PRICE_COLUMNS]
    df = df.dropna(subset=["close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# --- yfinance seam ----------------------------------------------------------

def _yfinance_download(
    tickers: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Single seam onto yfinance. Tests monkey-patch this to inject fixtures."""
    return yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )


def _fetch_ticker(
    ticker: str,
    start: date,
    end: date,
    breaker: CircuitBreaker,
    fetch_config,
) -> pd.DataFrame:
    """Fetch one ticker for [start, end] and return a normalized DataFrame."""
    end_exclusive = (end + timedelta(days=1)).isoformat()

    def _do() -> pd.DataFrame:
        try:
            raw = _yfinance_download(
                [ticker], start=start.isoformat(), end=end_exclusive,
            )
        except Exception as e:
            raise TransientError(f"yfinance error for {ticker}: {e}") from e
        if raw is None or raw.empty:
            raise PermanentError(f"yfinance returned no data for {ticker}")
        return raw

    raw = fetch_with_retry(_do, breaker, fetch_config=fetch_config, label=f"prices-{ticker}")
    return _normalize_yf_frame(raw, ticker)


# --- public API -------------------------------------------------------------

def ingest_prices(
    tickers: Optional[Sequence[str]] = None,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    force: bool = False,
    config: Optional[GraphSignalConfig] = None,
) -> IngestionSummary:
    """Fetch and persist daily OHLCV for `tickers` (default: full universe).

    Idempotent: a re-run only fetches dates after each ticker's last on-disk
    date. `force=True` refetches everything from `start_date`.
    """
    cfg = config or get_config()
    if tickers is None:
        tickers = load_universe("all")
    tickers = unique_ordered(tickers)
    eff_start = start_date or DEFAULT_START_DATE
    eff_end = end_date or date.today()
    breaker = CircuitBreaker(cfg.fetch.circuit_breaker_consecutive_failures)
    summary = IngestionSummary()

    for ticker in tickers:
        try:
            breaker.check()
        except CircuitOpenError:
            summary.circuit_open = True
            log.error(
                "circuit open after %d consecutive failures; halting",
                breaker.consecutive_failures,
            )
            break

        path = _parquet_path(cfg, ticker)
        existing_last = None if force else _existing_last_date(path)

        if existing_last is not None and existing_last >= eff_end:
            summary.already_current.append(ticker)
            continue

        fetch_start = (
            existing_last + timedelta(days=1) if existing_last else eff_start
        )

        try:
            new_df = _fetch_ticker(ticker, fetch_start, eff_end, breaker, cfg.fetch)
        except PermanentError as e:
            summary.skipped.append((ticker, str(e)))
            log.warning("skipping %s: %s", ticker, e)
            continue
        except TransientError as e:
            summary.skipped.append((ticker, f"gave up after retries: {e}"))
            log.error("gave up on %s: %s", ticker, e)
            continue
        except CircuitOpenError:
            summary.circuit_open = True
            log.error("circuit opened mid-run; halting")
            break

        if existing_last is not None and path.exists():
            existing_df = _read_parquet(path)
            combined = (
                pd.concat([existing_df, new_df], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            combined = new_df

        if combined.empty:
            summary.skipped.append((ticker, "empty after merge"))
            continue

        _write_parquet_atomic(path, combined)
        summary.updated.append(ticker)
        log.info(
            "ingested %s: %d rows (through %s)",
            ticker, len(combined),
            pd.to_datetime(combined["date"]).max().date(),
        )

    return summary


def load_prices(
    tickers: Union[Sequence[str], str],
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    adjusted: bool = True,
    *,
    config: Optional[GraphSignalConfig] = None,
) -> LoadedPrices:
    """Read prices for `tickers` from disk into a long-form DataFrame.

    With `adjusted=True` (default), OHLC are scaled by `adj_close / close` so
    the series is split- and dividend-adjusted. With `adjusted=False`, raw
    OHLC are returned alongside `adj_close` for callers that need both
    (e.g., historical dollar-volume calculations).

    Tickers that are not present on disk, or have no data in [start, end],
    are listed in the returned `missing` field.
    """
    cfg = config or get_config()
    if isinstance(tickers, str):
        tickers = [tickers]

    start_ts = pd.to_datetime(start) if start is not None else None
    end_ts = pd.to_datetime(end) if end is not None else None

    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for ticker in tickers:
        path = _parquet_path(cfg, ticker)
        if not path.exists():
            missing.append(ticker)
            continue
        df = _read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        if start_ts is not None:
            df = df[df["date"] >= start_ts]
        if end_ts is not None:
            df = df[df["date"] <= end_ts]
        if df.empty:
            missing.append(ticker)
            continue
        df = df.copy()
        df["ticker"] = ticker
        frames.append(df[["ticker", *PRICE_COLUMNS]])

    if not frames:
        out = pd.DataFrame(columns=["ticker", *PRICE_COLUMNS])
    else:
        out = pd.concat(frames, ignore_index=True)

    if adjusted and not out.empty:
        ratio = (out["adj_close"] / out["close"].where(out["close"] != 0, pd.NA)).fillna(1.0)
        for col in ("open", "high", "low", "close"):
            out[col] = out[col] * ratio

    return LoadedPrices(df=out, missing=missing)


def main() -> None:
    """Run a full universe ingestion. Wired up properly in Step 5."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    s = ingest_prices()
    log.info(
        "Done: %d updated, %d already-current, %d skipped, circuit_open=%s",
        len(s.updated), len(s.already_current), len(s.skipped), s.circuit_open,
    )


if __name__ == "__main__":
    main()
