# GraphSignal

Graph-based financial signal flagging engine (research prototype).

See [`MDs/README.md`](MDs/README.md) for project overview and goals.
See [`MDs/Phase0.md`](MDs/Phase0.md) for the data layer specification.

## Status

**Phase 0 — Data Layer** (complete).

Implemented:
- Smart-throttled fetch primitives (`graphsignal.data._fetch`).
- Universe snapshots (S&P 500 from iShares IVV, top-50 ETFs by yfinance volume).
- Price ingestion + loader (`graphsignal.data.prices`).
- Reference data: sectors, ETF metadata/inception dates, ETF constituents (`graphsignal.data.reference`).
- Event metadata: earnings, drawdowns, gaps, volume spikes (`graphsignal.data.events`).
- Single-command ingestion (`python -m graphsignal.data.ingest`).

Phase 0 is complete. Later phases should consume data only through `graphsignal.data`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

## Reproduce The Data Store

From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m graphsignal.data.ingest
```

The first full run downloads daily prices for the committed universe, then
builds reference snapshots and event metadata. It can take a while and may be
affected by yfinance throttling. Runs are checkpointed, so rerunning the command
picks up existing files instead of starting over.

For a quick subset run:

```bash
python -m graphsignal.data.ingest --tickers AAPL MSFT SPY --skip-earnings
```

To write somewhere other than `./data`:

```bash
python -m graphsignal.data.ingest --data-dir /tmp/graphsignal-data
```

To verify idempotency, run ingestion twice and compare manifests:

```bash
python -m graphsignal.data.ingest --manifest /tmp/graphsignal-manifest-1.csv
python -m graphsignal.data.ingest --manifest /tmp/graphsignal-manifest-2.csv
diff /tmp/graphsignal-manifest-1.csv /tmp/graphsignal-manifest-2.csv
```

The universe snapshots are committed and are not refreshed by this command.
Refreshing them is an explicit maintenance step:

```bash
python -m graphsignal.data.universe._build
```

## Data Sources

| Layer    | Source                                           | Notes |
| -------- | ------------------------------------------------ | ----- |
| S&P 500           | iShares IVV holdings CSV (BlackRock, public)        | Fixed snapshot, committed. |
| ETFs              | Curated candidate list, ranked by yfinance 30d ADV  | Top 50, committed snapshot. |
| Prices            | yfinance daily OHLCV + adjusted close               | Local Parquet store. |
| Sectors           | yfinance `.info` per ticker                         | Local Parquet snapshot. |
| ETF metadata      | yfinance `.info`                                    | Inception dates are best-effort. |
| ETF constituents  | yfinance `funds_data.top_holdings`                  | Top ~10–25 names per ETF. |
| Earnings events   | yfinance earnings dates                             | Best-effort; incomplete historical coverage. |
| Derived events    | Local price files                                   | Drawdowns, gaps, volume spikes. |

To refresh universe snapshots from current data:

```bash
python -m graphsignal.data.universe._build
```

To refresh reference data (sectors, ETF metadata, ETF constituents) from current data:

```bash
python -m graphsignal.data.reference
```

To refresh event metadata (earnings + derived price events):

```bash
python -m graphsignal.data.events
```

To run the full Phase 0 ingestion workflow:

```bash
python -m graphsignal.data.ingest
```

## Data Schemas

### `load_prices(...) -> LoadedPrices`

`LoadedPrices` is `(df, missing)`. Long-form DataFrame with one row per (ticker, date):

| column     | dtype           | meaning                                              |
| ---------- | --------------- | ---------------------------------------------------- |
| ticker     | str             | yfinance-style symbol (e.g., `AAPL`, `BRK-B`).       |
| date       | datetime64[ns]  | Trading day.                                         |
| open       | float64         | Open price (adjusted if `adjusted=True`, else raw).  |
| high       | float64         | High price (adjusted if `adjusted=True`, else raw).  |
| low        | float64         | Low price (adjusted if `adjusted=True`, else raw).   |
| close      | float64         | Close price (adjusted if `adjusted=True`, else raw). |
| volume     | int64           | Daily volume (always raw).                           |
| adj_close  | float64         | Split- and dividend-adjusted close.                  |

`missing` lists tickers requested but not present on disk (or empty in the date range).

### `load_universe(kind="all") -> list[str]`

Returns ticker symbols. `kind` is one of `"all"`, `"sp500"`, `"etfs"`.

### `load_sp500() -> DataFrame`

| ticker | name | sector | weight_pct | source | retrieved_at |

### `load_etfs() -> DataFrame`

| ticker | name | category | avg_dollar_volume | rank | source | retrieved_at |

### `load_sectors() -> DataFrame`

| ticker | sector | industry | quote_type | source | retrieved_at |

`sector` and `industry` are null for ETFs and a handful of edge-case tickers.

### `load_etf_constituents(etf) -> list[str]` / `load_etf_holdings(etf) -> DataFrame`

`load_etf_constituents` returns just constituent tickers in weight order.
`load_etf_holdings` returns the full DataFrame:

| etf | ticker | name | weight_pct | source | retrieved_at |

Coverage is the top ~10–25 names per ETF (yfinance limitation). Bond and
commodity ETFs without equity holdings are not stored.

### `load_etf_metadata() -> DataFrame`

| ticker | name | category | inception_date | source | retrieved_at |

`inception_date` is nullable because yfinance does not expose it consistently
for every fund.

### `load_events(event_type, tickers=None, start=None, end=None) -> DataFrame`

`event_type` is one of `"earnings"`, `"drawdowns"`, `"gaps"`,
`"volume_spikes"`, or `"all"`.

Common event columns:

| event_type | ticker | date | source | created_at |

Earnings add:

| event_time |

Drawdowns add:

| window_days | threshold_pct | drawdown_pct | peak_date |

Gaps add:

| threshold_pct | gap_pct | prior_close | open |

Volume spikes add:

| window_days | zscore_threshold | volume_zscore | volume | trailing_mean | trailing_std |

## Storage Layout

```
data/                           # gitignored, regenerated by ingestion
├── prices/
│   ├── AAPL.parquet            # one file per ticker, partitioned by ticker
│   ├── MSFT.parquet
│   └── ...
├── reference/
│   ├── sectors.parquet         # one row per ticker
│   ├── etf_metadata.parquet    # ETF inception dates where available
│   └── etf_constituents/
│       ├── SPY.parquet
│       └── ...
└── events/
    ├── earnings.parquet
    ├── drawdowns.parquet
    ├── gaps.parquet
    └── volume_spikes.parquet

graphsignal/data/universe/      # committed snapshots
├── sp500.csv
├── etfs.csv
└── etf_candidates.csv
```

## Configuration

Defaults live in `graphsignal/config.py`. To override, create `graphsignal.yaml` at the repo root:

```yaml
data_dir: ./data
fetch:
  max_retries: 5
  initial_backoff_seconds: 1.0
  max_backoff_seconds: 30.0
  concurrency: 3
  circuit_breaker_consecutive_failures: 10
events:
  drawdown_threshold_pct: 0.20
  drawdown_window_days: 60
  gap_threshold_pct: 0.05
  volume_spike_zscore: 3.0
  volume_spike_window_days: 20
```

## Known Limitations (Phase 0)

- **Survivorship bias.** Universe is fixed at a recent snapshot; delisted/acquired companies are absent.
- **Static ETF constituents.** ETF holdings are recent snapshots, not point-in-time history.
- **Static index membership.** S&P 500 list does not reflect historical changes.
- **Sparse early-period ETFs.** Many ETFs in the universe did not exist or had minimal AUM in the early 2000s; downstream code must filter on inception.
- **Incomplete earnings history.** yfinance earnings dates are best-effort and must not be treated as complete historical ground truth.
- **yfinance dependency.** Unofficial, subject to change; isolated to specific ingestion modules so it can be swapped without touching consumer code.
