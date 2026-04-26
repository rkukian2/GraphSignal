"""Centralized configuration for GraphSignal.

All paths, retry settings, and event thresholds flow through this module so
that behavior is reproducible from a single source. If `graphsignal.yaml`
exists at the repo root, it overrides the defaults below; otherwise the
defaults apply.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = REPO_ROOT / "graphsignal.yaml"


class FetchConfig(BaseModel):
    max_retries: int = 5
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    concurrency: int = 3
    circuit_breaker_consecutive_failures: int = 10


class EventThresholds(BaseModel):
    drawdown_threshold_pct: float = 0.20
    drawdown_window_days: int = 60
    gap_threshold_pct: float = 0.05
    volume_spike_zscore: float = 3.0
    volume_spike_window_days: int = 20


class GraphSignalConfig(BaseModel):
    data_dir: Path = REPO_ROOT / "data"
    universe_dir: Path = REPO_ROOT / "graphsignal" / "data" / "universe"
    fetch: FetchConfig = Field(default_factory=FetchConfig)
    events: EventThresholds = Field(default_factory=EventThresholds)


def get_config(config_path: Optional[Path] = None) -> GraphSignalConfig:
    """Load config from `graphsignal.yaml` at repo root if present, else defaults.

    Pass an explicit path for tests. Reads and parses on every call; this is
    cheap (small YAML) and avoids stale state between tests.
    """
    path = config_path or DEFAULT_CONFIG_FILE
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        return GraphSignalConfig.model_validate(raw)
    return GraphSignalConfig()
