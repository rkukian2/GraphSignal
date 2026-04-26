"""Small storage helpers shared by Phase 0 ingestion modules."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def unique_ordered(values: Sequence[str]) -> list[str]:
    """Return values without duplicates, preserving first-seen order."""
    return list(dict.fromkeys(values))


def read_parquet_or_empty(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    return pd.read_parquet(path)


def write_dataframe_atomic(
    path: Path,
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    date_columns: Sequence[str] = (),
) -> None:
    out = df[list(columns)].copy() if not df.empty else pd.DataFrame(columns=list(columns))
    for col in date_columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def missing_from_snapshot(
    requested: Sequence[str],
    existing: pd.DataFrame,
    *,
    key: str = "ticker",
) -> list[str]:
    if existing.empty:
        return list(requested)
    existing_values = set(existing[key].tolist())
    return [value for value in requested if value not in existing_values]


def merge_snapshot(
    existing: pd.DataFrame,
    new_df: pd.DataFrame,
    columns: Sequence[str],
    *,
    key_columns: Sequence[str],
    refreshed_values: Sequence[str] | None = None,
    refreshed_column: str = "ticker",
    sort_by: Sequence[str] = (),
) -> pd.DataFrame:
    base = existing
    if refreshed_values is not None and not existing.empty:
        base = existing[~existing[refreshed_column].isin(refreshed_values)]

    frames = [df for df in (base, new_df) if not df.empty]
    if not frames:
        return pd.DataFrame(columns=list(columns))

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=list(key_columns), keep="last")
    if sort_by:
        out = out.sort_values(list(sort_by))
    return out.reset_index(drop=True)[list(columns)]
