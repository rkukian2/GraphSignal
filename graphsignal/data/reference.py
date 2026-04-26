"""Reference data: sectors, ETF metadata, and ETF constituent lists.

Day-to-day code reads via `load_sectors()`, `load_etf_metadata()`, and
`load_etf_constituents(etf)`.
Refresh from external sources via:

    python -m graphsignal.data.reference

Sources:
- Sectors / industry: yfinance `.info` per ticker. ETFs do not have sectors;
  they are recorded with null sector/industry and `quote_type="ETF"`.
- ETF constituents: yfinance `funds_data.top_holdings`, which returns the top
  ~10-25 names per fund. Commodity/bond ETFs without equity holdings are
  logged and skipped — that's a known Phase 0 limitation.
- ETF inception dates: best-effort yfinance `.info` metadata.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

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
from graphsignal.data.universe import load_etfs, load_universe

log = logging.getLogger(__name__)

SECTORS_FILENAME = "sectors.parquet"
ETF_CONSTITUENTS_DIRNAME = "etf_constituents"
ETF_METADATA_FILENAME = "etf_metadata.parquet"

SECTORS_COLUMNS = ["ticker", "sector", "industry", "quote_type", "source", "retrieved_at"]
HOLDINGS_COLUMNS = ["etf", "ticker", "name", "weight_pct", "source", "retrieved_at"]
ETF_METADATA_COLUMNS = [
    "ticker", "name", "category", "inception_date", "source", "retrieved_at",
]


@dataclass
class ReferenceIngestionSummary:
    sectors_updated: list[str] = field(default_factory=list)
    sectors_skipped: list[tuple[str, str]] = field(default_factory=list)
    etfs_updated: list[str] = field(default_factory=list)
    etfs_skipped: list[tuple[str, str]] = field(default_factory=list)
    etf_metadata_updated: list[str] = field(default_factory=list)
    etf_metadata_skipped: list[tuple[str, str]] = field(default_factory=list)
    sectors_circuit_open: bool = False
    etfs_circuit_open: bool = False
    etf_metadata_circuit_open: bool = False


# --- paths ------------------------------------------------------------------

def _reference_dir(cfg: GraphSignalConfig) -> Path:
    return cfg.data_dir / "reference"


def _sectors_path(cfg: GraphSignalConfig) -> Path:
    return _reference_dir(cfg) / SECTORS_FILENAME


def _etf_constituents_dir(cfg: GraphSignalConfig) -> Path:
    return _reference_dir(cfg) / ETF_CONSTITUENTS_DIRNAME


def _etf_constituent_path(cfg: GraphSignalConfig, etf: str) -> Path:
    return _etf_constituents_dir(cfg) / f"{etf}.parquet"


def _etf_metadata_path(cfg: GraphSignalConfig) -> Path:
    return _reference_dir(cfg) / ETF_METADATA_FILENAME


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# --- yfinance seams ---------------------------------------------------------

def _yfinance_info(ticker: str) -> dict:
    """Fetch yfinance metadata for one ticker. Test seam — monkey-patched."""
    return yf.Ticker(ticker).info or {}


def _yfinance_top_holdings(etf: str) -> pd.DataFrame:
    """Fetch ETF top holdings. Test seam — monkey-patched.

    Returns a DataFrame indexed by Symbol with columns Name and Holding Percent
    (the latter as a fraction, e.g. 0.075 = 7.5%). Raises PermanentError if the
    ticker is not a fund or has no equity holdings.
    """
    t = yf.Ticker(etf)
    funds = t.funds_data
    if funds is None:
        raise PermanentError(f"{etf}: not a fund (no funds_data)")
    df = funds.top_holdings
    if df is None or df.empty:
        raise PermanentError(f"{etf}: no top holdings reported")
    return df


# --- sectors ----------------------------------------------------------------

def _extract_sector_row(ticker: str, info: dict) -> dict:
    """Pull sector/industry/quote_type out of a yfinance .info dict."""
    return {
        "ticker": ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "quote_type": info.get("quoteType"),
    }


def ingest_sectors(
    tickers: Optional[Sequence[str]] = None,
    *,
    force: bool = False,
    config: Optional[GraphSignalConfig] = None,
) -> tuple[list[str], list[tuple[str, str]], bool]:
    """Fetch sector/industry from yfinance for each ticker and persist as a
    single Parquet snapshot. Returns (updated, skipped, circuit_open).

    Tickers without sector data (ETFs and a handful of edge cases) are still
    recorded — with null sector/industry — so downstream code sees them.
    """
    cfg = config or get_config()
    if tickers is None:
        tickers = load_universe("all")
    tickers = unique_ordered(tickers)
    path = _sectors_path(cfg)
    existing = read_parquet_or_empty(path, SECTORS_COLUMNS)
    fetch_tickers = tickers if force else missing_from_snapshot(tickers, existing)

    if not fetch_tickers:
        log.info("sector snapshot already has %d requested tickers", len(tickers))
        return [], [], False

    breaker = CircuitBreaker(cfg.fetch.circuit_breaker_consecutive_failures)
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    circuit_open = False

    for ticker in fetch_tickers:
        try:
            breaker.check()
        except CircuitOpenError:
            circuit_open = True
            log.error("circuit open after %d failures; halting sectors ingestion",
                      breaker.consecutive_failures)
            break

        def _do() -> dict:
            try:
                info = _yfinance_info(ticker)
            except Exception as e:
                raise TransientError(f"{ticker}: yfinance.info: {e}") from e
            return info

        try:
            info = fetch_with_retry(_do, breaker, fetch_config=cfg.fetch, label=f"info-{ticker}")
        except (PermanentError, TransientError) as e:
            skipped.append((ticker, str(e)))
            log.warning("skipping sector for %s: %s", ticker, e)
            continue
        except CircuitOpenError:
            circuit_open = True
            break

        rows.append(_extract_sector_row(ticker, info))

    if rows:
        new_df = pd.DataFrame(rows)
        new_df["source"] = "yfinance .info"
        new_df["retrieved_at"] = _today_iso()
        new_df = new_df[SECTORS_COLUMNS]
        df = merge_snapshot(
            existing,
            new_df,
            SECTORS_COLUMNS,
            key_columns=["ticker"],
            refreshed_values=tickers if force else None,
            sort_by=["ticker"],
        )
        write_dataframe_atomic(path, df, SECTORS_COLUMNS)
        log.info("wrote %d sector rows to %s", len(df), path)

    return [r["ticker"] for r in rows], skipped, circuit_open


def load_sectors(*, config: Optional[GraphSignalConfig] = None) -> pd.DataFrame:
    """Return the sector/industry snapshot DataFrame.

    Columns: ticker, sector, industry, quote_type, source, retrieved_at.
    Tickers without sector data have null sector/industry.
    """
    cfg = config or get_config()
    path = _sectors_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m graphsignal.data.reference"
        )
    return pd.read_parquet(path)


# --- ETF constituents -------------------------------------------------------

def _normalize_holdings(df: pd.DataFrame, etf: str) -> pd.DataFrame:
    """Convert yfinance funds_data.top_holdings into our schema.

    Input: DataFrame indexed by Symbol with columns Name, Holding Percent
    (where the percent is a fraction in [0, 1]).
    Output: long-form rows with columns matching HOLDINGS_COLUMNS.
    """
    out = df.copy()
    if out.index.name == "Symbol":
        out = out.reset_index()
    if "Symbol" in out.columns:
        out = out.rename(columns={"Symbol": "ticker"})
    elif "ticker" not in out.columns:
        raise PermanentError(f"{etf}: top_holdings missing Symbol/ticker column")

    if "Name" in out.columns:
        out = out.rename(columns={"Name": "name"})
    elif "name" not in out.columns:
        out["name"] = pd.NA

    weight_col = next(
        (c for c in out.columns if "percent" in c.lower() or "weight" in c.lower()),
        None,
    )
    if weight_col is None:
        out["weight_pct"] = pd.NA
    else:
        out["weight_pct"] = pd.to_numeric(out[weight_col], errors="coerce") * 100.0

    out["etf"] = etf
    out["source"] = "yfinance funds_data.top_holdings"
    out["retrieved_at"] = _today_iso()
    return out[HOLDINGS_COLUMNS].reset_index(drop=True)


def ingest_etf_constituents(
    etfs: Optional[Sequence[str]] = None,
    *,
    force: bool = False,
    config: Optional[GraphSignalConfig] = None,
) -> tuple[list[str], list[tuple[str, str]], bool]:
    """Fetch top-holdings per ETF and persist one Parquet per ETF.
    Returns (updated, skipped, circuit_open)."""
    cfg = config or get_config()
    if etfs is None:
        etfs = load_universe("etfs")
    etfs = unique_ordered(etfs)
    breaker = CircuitBreaker(cfg.fetch.circuit_breaker_consecutive_failures)
    updated: list[str] = []
    skipped: list[tuple[str, str]] = []
    circuit_open = False

    for etf in etfs:
        path = _etf_constituent_path(cfg, etf)
        if path.exists() and not force:
            continue

        try:
            breaker.check()
        except CircuitOpenError:
            circuit_open = True
            log.error("circuit open; halting ETF constituents ingestion")
            break

        def _do() -> pd.DataFrame:
            try:
                return _yfinance_top_holdings(etf)
            except PermanentError:
                raise
            except Exception as e:
                raise TransientError(f"{etf}: top_holdings: {e}") from e

        try:
            holdings = fetch_with_retry(_do, breaker, fetch_config=cfg.fetch, label=f"holdings-{etf}")
        except (PermanentError, TransientError) as e:
            skipped.append((etf, str(e)))
            log.warning("skipping holdings for %s: %s", etf, e)
            continue
        except CircuitOpenError:
            circuit_open = True
            break

        try:
            normalized = _normalize_holdings(holdings, etf)
        except PermanentError as e:
            skipped.append((etf, str(e)))
            log.warning("skipping holdings for %s: %s", etf, e)
            continue

        write_dataframe_atomic(path, normalized, HOLDINGS_COLUMNS)
        updated.append(etf)
        log.info("ingested %s holdings: %d names", etf, len(normalized))

    return updated, skipped, circuit_open


def load_etf_holdings(
    etf: str,
    *,
    config: Optional[GraphSignalConfig] = None,
) -> pd.DataFrame:
    """Return the full ETF holdings DataFrame for `etf`.

    Columns: etf, ticker, name, weight_pct, source, retrieved_at.
    """
    cfg = config or get_config()
    path = _etf_constituent_path(cfg, etf)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m graphsignal.data.reference"
        )
    return pd.read_parquet(path)


def load_etf_constituents(
    etf: str,
    *,
    config: Optional[GraphSignalConfig] = None,
) -> list[str]:
    """Return just the constituent tickers for `etf`, in weight order."""
    return load_etf_holdings(etf, config=config)["ticker"].tolist()


# --- ETF metadata / inception dates -----------------------------------------

def _coerce_epoch_date(value) -> Optional[str]:
    if value in (None, "", 0):
        return None
    try:
        ts = pd.to_datetime(int(value), unit="s", utc=True)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def _extract_etf_metadata_row(etf: str, info: dict, universe_row: Optional[pd.Series] = None) -> dict:
    """Build the ETF metadata row, including best-effort inception date."""
    name = None
    category = None
    if universe_row is not None:
        name = universe_row.get("name")
        category = universe_row.get("category")

    inception = (
        _coerce_epoch_date(info.get("fundInceptionDate"))
        or _coerce_epoch_date(info.get("firstTradeDateEpochUtc"))
    )
    return {
        "ticker": etf,
        "name": info.get("longName") or info.get("shortName") or name,
        "category": category,
        "inception_date": inception,
        "source": "yfinance .info",
        "retrieved_at": _today_iso(),
    }


def _etf_universe_rows(etfs: Sequence[str]) -> dict[str, pd.Series]:
    try:
        df = load_etfs()
    except FileNotFoundError:
        return {}
    wanted = set(etfs)
    return {
        row["ticker"]: row
        for _, row in df[df["ticker"].isin(wanted)].iterrows()
    }


def ingest_etf_metadata(
    etfs: Optional[Sequence[str]] = None,
    *,
    force: bool = False,
    config: Optional[GraphSignalConfig] = None,
) -> tuple[list[str], list[tuple[str, str]], bool]:
    """Fetch ETF metadata and persist a snapshot with inception dates.

    Existing rows are checkpoints. By default, only ETFs missing from the
    snapshot are fetched; `force=True` refreshes the requested ETFs.
    """
    cfg = config or get_config()
    if etfs is None:
        etfs = load_universe("etfs")
    etfs = unique_ordered(etfs)
    path = _etf_metadata_path(cfg)
    existing = read_parquet_or_empty(path, ETF_METADATA_COLUMNS)
    fetch_etfs = etfs if force else missing_from_snapshot(etfs, existing)

    if not fetch_etfs:
        log.info("ETF metadata snapshot already has %d requested ETFs", len(etfs))
        return [], [], False

    universe_rows = _etf_universe_rows(fetch_etfs)
    breaker = CircuitBreaker(cfg.fetch.circuit_breaker_consecutive_failures)
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    circuit_open = False

    for etf in fetch_etfs:
        try:
            breaker.check()
        except CircuitOpenError:
            circuit_open = True
            log.error("circuit open; halting ETF metadata ingestion")
            break

        def _do() -> dict:
            try:
                return _yfinance_info(etf)
            except Exception as e:
                raise TransientError(f"{etf}: yfinance.info: {e}") from e

        try:
            info = fetch_with_retry(_do, breaker, fetch_config=cfg.fetch, label=f"etf-info-{etf}")
        except (PermanentError, TransientError) as e:
            skipped.append((etf, str(e)))
            log.warning("ETF metadata unavailable for %s: %s", etf, e)
            info = {}
        except CircuitOpenError:
            circuit_open = True
            break

        rows.append(_extract_etf_metadata_row(etf, info, universe_rows.get(etf)))

    if rows:
        new_df = pd.DataFrame(rows)[ETF_METADATA_COLUMNS]
        df = merge_snapshot(
            existing,
            new_df,
            ETF_METADATA_COLUMNS,
            key_columns=["ticker"],
            refreshed_values=etfs if force else None,
            sort_by=["ticker"],
        )
        write_dataframe_atomic(path, df, ETF_METADATA_COLUMNS)
        log.info("wrote %d ETF metadata rows to %s", len(df), path)

    return [r["ticker"] for r in rows], skipped, circuit_open


def load_etf_metadata(*, config: Optional[GraphSignalConfig] = None) -> pd.DataFrame:
    """Return ETF metadata, including inception_date where yfinance provides it."""
    cfg = config or get_config()
    path = _etf_metadata_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m graphsignal.data.reference"
        )
    return pd.read_parquet(path)


def ingest_reference(
    *,
    force: bool = False,
    config: Optional[GraphSignalConfig] = None,
) -> ReferenceIngestionSummary:
    """Run all reference-data ingestion steps."""
    cfg = config or get_config()
    summary = ReferenceIngestionSummary()
    (
        summary.sectors_updated,
        summary.sectors_skipped,
        summary.sectors_circuit_open,
    ) = ingest_sectors(force=force, config=cfg)
    (
        summary.etf_metadata_updated,
        summary.etf_metadata_skipped,
        summary.etf_metadata_circuit_open,
    ) = ingest_etf_metadata(force=force, config=cfg)
    (
        summary.etfs_updated,
        summary.etfs_skipped,
        summary.etfs_circuit_open,
    ) = ingest_etf_constituents(force=force, config=cfg)
    return summary


# --- CLI --------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = ingest_reference()
    log.info(
        "sectors: %d updated, %d skipped, circuit_open=%s",
        len(summary.sectors_updated),
        len(summary.sectors_skipped),
        summary.sectors_circuit_open,
    )
    log.info(
        "etf metadata: %d updated, %d skipped, circuit_open=%s",
        len(summary.etf_metadata_updated),
        len(summary.etf_metadata_skipped),
        summary.etf_metadata_circuit_open,
    )
    log.info(
        "etf constituents: %d updated, %d skipped, circuit_open=%s",
        len(summary.etfs_updated),
        len(summary.etfs_skipped),
        summary.etfs_circuit_open,
    )


if __name__ == "__main__":
    main()
