"""Refresh universe snapshots. Run via:

    python -m graphsignal.data.universe._build

Hits external sources and overwrites the committed snapshot files. Intended as
a maintenance script run only when explicitly refreshing the universe (e.g.,
once at project start, or periodically by a maintainer). Day-to-day code
reads the snapshots via load_universe().

Sources:
- S&P 500 list: iShares IVV holdings CSV (BlackRock, public download).
- ETFs:        ranked by 30-day average dollar volume from yfinance, drawn from
               the curated candidate list at etf_candidates.csv. Top 50 saved.
"""
from __future__ import annotations

import io
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from graphsignal.data._fetch import (
    PermanentError,
    TransientError,
    fetch_with_retry,
)
from graphsignal.data.universe import (
    ETF_CANDIDATES_FILE,
    ETFS_FILE,
    SP500_FILE,
)

IVV_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239726/"
    "ishares-core-sp-500-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IVV_holdings&dataType=fund"
)
IVV_SOURCE_LABEL = "iShares IVV holdings"
ETF_SOURCE_LABEL = "yfinance 30d avg dollar volume"
TOP_N_ETFS = 50
ETF_RANKING_DAYS = 30

# Valid yfinance-style tickers: 1-6 chars, A-Z/0-9/dash, must start alnum.
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{0,5}$")

log = logging.getLogger(__name__)


def _normalize_ticker(t: str) -> str:
    """yfinance uses '-' where issuers use '.' (BRK.B -> BRK-B)."""
    return t.replace(".", "-").upper().strip()


def _is_valid_ticker(t: str) -> bool:
    """Reject placeholders ('-', empty) and malformed strings."""
    return bool(TICKER_RE.match(t))


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def fetch_ivv_holdings_csv() -> bytes:
    def _do() -> bytes:
        try:
            r = requests.get(
                IVV_HOLDINGS_URL,
                headers={"User-Agent": "Mozilla/5.0 GraphSignal universe-build"},
                timeout=30,
            )
        except (requests.ConnectionError, requests.Timeout) as e:
            raise TransientError(f"network error fetching IVV holdings: {e}") from e
        if r.status_code in (429, 500, 502, 503, 504):
            raise TransientError(f"HTTP {r.status_code} from iShares")
        if not r.ok:
            raise PermanentError(f"HTTP {r.status_code} from iShares")
        return r.content

    return fetch_with_retry(_do, label="ivv-holdings")


def parse_ivv_csv(raw: bytes) -> pd.DataFrame:
    """Parse iShares IVV holdings CSV into a clean equities-only DataFrame.

    The file has a metadata prelude before the table. We locate the header row
    dynamically rather than skipping a fixed number of lines, so the parser
    survives small format changes.
    """
    text = raw.decode("utf-8-sig")
    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("Ticker,")),
        None,
    )
    if header_idx is None:
        raise PermanentError("IVV CSV: 'Ticker,' header row not found")

    table = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(table), on_bad_lines="skip")
    df = df[df["Asset Class"] == "Equity"].copy()
    df = df[df["Ticker"].notna()]
    df["ticker"] = df["Ticker"].apply(_normalize_ticker)
    valid = df["ticker"].apply(_is_valid_ticker)
    if (~valid).any():
        log.info(
            "Dropping %d non-ticker rows from IVV holdings (e.g. placeholders): %s",
            (~valid).sum(),
            df.loc[~valid, "ticker"].tolist()[:5],
        )
    df = df[valid]
    df = df[["ticker", "Name", "Sector", "Weight (%)"]]
    df.columns = ["ticker", "name", "sector", "weight_pct"]
    df["source"] = IVV_SOURCE_LABEL
    df["retrieved_at"] = _today()
    return df.sort_values("ticker").reset_index(drop=True)


def rank_etfs_by_volume(candidates: pd.DataFrame, days: int = ETF_RANKING_DAYS) -> pd.DataFrame:
    """Fetch recent OHLCV for all candidate tickers in one batch and rank by
    average daily dollar volume (close * volume) over the trailing `days`.

    yfinance occasionally returns empty data for a ticker; those are logged
    and excluded from the ranking rather than crashing the run.
    """
    tickers = candidates["ticker"].tolist()
    log.info("Fetching %d candidate ETFs for volume ranking...", len(tickers))

    raw = yf.download(
        tickers,
        period=f"{days * 3}d",  # generous window to cover holidays / partial data
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    rows = []
    for ticker in tickers:
        try:
            sub = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            log.warning("no data returned for %s; skipping", ticker)
            continue
        if sub.empty or "Close" not in sub.columns or "Volume" not in sub.columns:
            log.warning("no OHLCV for %s; skipping", ticker)
            continue
        recent = sub.dropna(subset=["Close", "Volume"]).tail(days)
        if recent.empty:
            log.warning("insufficient data for %s; skipping", ticker)
            continue
        avg_dv = float((recent["Close"] * recent["Volume"]).mean())
        rows.append({"ticker": ticker, "avg_dollar_volume": avg_dv})

    ranked = pd.DataFrame(rows).sort_values("avg_dollar_volume", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    enriched = ranked.merge(candidates, on="ticker", how="left")
    enriched = enriched[["ticker", "name", "category", "avg_dollar_volume", "rank"]]
    enriched["source"] = ETF_SOURCE_LABEL
    enriched["retrieved_at"] = _today()
    return enriched


def build_sp500() -> Path:
    log.info("Downloading iShares IVV holdings...")
    raw = fetch_ivv_holdings_csv()
    df = parse_ivv_csv(raw)
    log.info("Parsed %d S&P 500 equities", len(df))
    SP500_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SP500_FILE, index=False)
    log.info("Wrote %s", SP500_FILE)
    return SP500_FILE


def build_etfs(top_n: int = TOP_N_ETFS) -> Path:
    candidates = pd.read_csv(ETF_CANDIDATES_FILE)
    ranked = rank_etfs_by_volume(candidates)
    top = ranked.head(top_n).reset_index(drop=True)
    top["rank"] = top.index + 1
    log.info("Selected top %d ETFs by 30d avg dollar volume", len(top))
    ETFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(ETFS_FILE, index=False)
    log.info("Wrote %s", ETFS_FILE)
    return ETFS_FILE


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_sp500()
    build_etfs()


if __name__ == "__main__":
    sys.exit(main())
