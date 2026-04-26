"""Universe snapshots: the fixed list of tickers GraphSignal works with.

Day-to-day code reads via `load_universe()`. The snapshot CSVs (`sp500.csv`,
`etfs.csv`) are committed to the repo. To refresh them from external sources,
run:

    python -m graphsignal.data.universe._build

The committed candidate file `etf_candidates.csv` is the auditable seed list
used by the refresh script when ranking ETFs by trading volume.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

UNIVERSE_DIR = Path(__file__).resolve().parent
SP500_FILE = UNIVERSE_DIR / "sp500.csv"
ETFS_FILE = UNIVERSE_DIR / "etfs.csv"
ETF_CANDIDATES_FILE = UNIVERSE_DIR / "etf_candidates.csv"

UniverseKind = Literal["all", "sp500", "etfs"]


class UniverseSnapshotMissing(FileNotFoundError):
    """Raised when a required universe snapshot file is not present.

    Run `python -m graphsignal.data.universe._build` to generate it.
    """


def _read_or_raise(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise UniverseSnapshotMissing(
            f"{path.name} not found at {path}. "
            f"Run: python -m graphsignal.data.universe._build"
        )
    return pd.read_csv(path)


def load_sp500() -> pd.DataFrame:
    """Return the S&P 500 snapshot as a DataFrame.

    Columns: ticker, name, sector, weight_pct, source, retrieved_at.
    """
    return _read_or_raise(SP500_FILE)


def load_etfs() -> pd.DataFrame:
    """Return the ETF snapshot (top 50 by volume) as a DataFrame.

    Columns: ticker, name, category, avg_dollar_volume, rank, source, retrieved_at.
    """
    return _read_or_raise(ETFS_FILE)


def load_universe(
    kind: UniverseKind = "all",
    as_of: Optional[str] = None,
) -> list[str]:
    """Return the list of tickers in the universe.

    - kind="all": S&P 500 + ETFs (default)
    - kind="sp500": equities only
    - kind="etfs": ETFs only
    - as_of: reserved for future point-in-time support; currently must be None.
    """
    if as_of is not None:
        raise NotImplementedError(
            "Point-in-time universe is out of scope for Phase 0; pass as_of=None."
        )
    if kind == "sp500":
        return load_sp500()["ticker"].tolist()
    if kind == "etfs":
        return load_etfs()["ticker"].tolist()
    if kind == "all":
        return load_sp500()["ticker"].tolist() + load_etfs()["ticker"].tolist()
    raise ValueError(f"unknown kind: {kind!r}")
