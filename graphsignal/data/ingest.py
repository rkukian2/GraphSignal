"""Single-command Phase 0 data-store ingestion.

Run from a fresh clone with:

    python -m graphsignal.data.ingest

The committed universe snapshots are the source of truth. This command does
not rebuild `sp500.csv` or `etfs.csv`.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from graphsignal.config import GraphSignalConfig, get_config
from graphsignal.data import events, prices, reference
from graphsignal.data._storage import unique_ordered
from graphsignal.data.events import EventIngestionSummary
from graphsignal.data.prices import IngestionSummary
from graphsignal.data.reference import ReferenceIngestionSummary
from graphsignal.data.universe import load_universe

log = logging.getLogger(__name__)


@dataclass
class DataStoreIngestionSummary:
    prices: IngestionSummary
    reference: ReferenceIngestionSummary
    events: EventIngestionSummary


def _with_data_dir(cfg: GraphSignalConfig, data_dir: Optional[Path]) -> GraphSignalConfig:
    if data_dir is None:
        return cfg
    return cfg.model_copy(update={"data_dir": data_dir})


def _parse_date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    return date.fromisoformat(value)


def data_store_manifest(data_dir: Path) -> pd.DataFrame:
    """Return a deterministic manifest of files under `data_dir`.

    Columns: path, size_bytes, sha256. Paths are relative to `data_dir`.
    """
    rows: list[dict] = []
    if not data_dir.exists():
        return pd.DataFrame(columns=["path", "size_bytes", "sha256"])
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "path": path.relative_to(data_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest,
        })
    return pd.DataFrame(rows, columns=["path", "size_bytes", "sha256"])


def _reference_for_tickers(
    tickers: Sequence[str],
    *,
    force: bool,
    config: GraphSignalConfig,
) -> ReferenceIngestionSummary:
    etf_universe = set(load_universe("etfs"))
    etfs = [t for t in tickers if t in etf_universe]
    summary = ReferenceIngestionSummary()
    (
        summary.sectors_updated,
        summary.sectors_skipped,
        summary.sectors_circuit_open,
    ) = reference.ingest_sectors(tickers, force=force, config=config)
    (
        summary.etf_metadata_updated,
        summary.etf_metadata_skipped,
        summary.etf_metadata_circuit_open,
    ) = reference.ingest_etf_metadata(etfs, force=force, config=config)
    (
        summary.etfs_updated,
        summary.etfs_skipped,
        summary.etfs_circuit_open,
    ) = reference.ingest_etf_constituents(etfs, force=force, config=config)
    return summary


def ingest_all(
    tickers: Optional[Sequence[str]] = None,
    *,
    force: bool = False,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    skip_earnings: bool = False,
    config: Optional[GraphSignalConfig] = None,
) -> DataStoreIngestionSummary:
    """Run Phase 0 ingestion in dependency order: prices, reference, events."""
    cfg = config or get_config()
    active_tickers = unique_ordered(tickers or load_universe("all"))

    log.info("ingesting prices for %d tickers", len(active_tickers))
    price_summary = prices.ingest_prices(
        active_tickers,
        start_date=start_date,
        end_date=end_date,
        force=force,
        config=cfg,
    )

    log.info("ingesting reference data")
    if tickers is None:
        reference_summary = reference.ingest_reference(force=force, config=cfg)
    else:
        reference_summary = _reference_for_tickers(active_tickers, force=force, config=cfg)

    log.info("ingesting events")
    if skip_earnings:
        event_summary = EventIngestionSummary()
        (
            event_summary.derived_updated,
            event_summary.derived_missing_prices,
        ) = events.ingest_derived_events(active_tickers, force=force, config=cfg)
    else:
        event_summary = events.ingest_events(active_tickers, force=force, config=cfg)

    return DataStoreIngestionSummary(
        prices=price_summary,
        reference=reference_summary,
        events=event_summary,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Phase 0 local data store.")
    parser.add_argument("--force", action="store_true", help="Refresh existing local files.")
    parser.add_argument("--tickers", nargs="+", help="Restrict ingestion to these tickers.")
    parser.add_argument("--start-date", help="Price ingestion start date: YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Price ingestion end date: YYYY-MM-DD.")
    parser.add_argument("--skip-earnings", action="store_true", help="Skip best-effort yfinance earnings fetches.")
    parser.add_argument("--data-dir", type=Path, help="Override the configured data directory.")
    parser.add_argument("--manifest", type=Path, help="Write a CSV file manifest after ingestion.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> DataStoreIngestionSummary:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = _with_data_dir(get_config(), args.data_dir)
    summary = ingest_all(
        tickers=args.tickers,
        force=args.force,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        skip_earnings=args.skip_earnings,
        config=cfg,
    )
    log.info(
        "prices: %d updated, %d already-current, %d skipped, circuit_open=%s",
        len(summary.prices.updated),
        len(summary.prices.already_current),
        len(summary.prices.skipped),
        summary.prices.circuit_open,
    )
    log.info(
        "reference: sectors=%d, etf_metadata=%d, etf_holdings=%d",
        len(summary.reference.sectors_updated),
        len(summary.reference.etf_metadata_updated),
        len(summary.reference.etfs_updated),
    )
    log.info(
        "events: earnings=%d, derived=%s",
        len(summary.events.earnings_updated),
        ",".join(summary.events.derived_updated),
    )
    if args.manifest is not None:
        manifest = data_store_manifest(cfg.data_dir)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.manifest, index=False)
        log.info("wrote manifest to %s", args.manifest)
    return summary


if __name__ == "__main__":
    main()
