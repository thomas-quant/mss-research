# MSS Research

Event-study tools for testing whether retail trading features predict near-term market direction:

- market structure shifts (MSS)
- RSI divergences
- volume divergences
- MSS break parameters: trade-through vs close-through, momentum, relative volume, short-term vs intermediate-term structure

Raw data and large generated output tables are intentionally ignored by git. Lightweight matplotlib figures in `outputs/figures/` are tracked so the README can show current research snapshots.

## Data

Expected local files:

```text
data/es_1m.parquet
data/nq_1m.parquet
```

Required columns:

```text
datetime_utc, Open, High, Low, Close, Volume
```

## Run study

```bash
PYTHONPATH=src python3 -m mss_research run --data data --out outputs --plots
```

Useful faster run:

```bash
PYTHONPATH=src python3 -m mss_research run --data data --out outputs --timeframes 15min --horizons 5 --bootstrap 100 --plots
```

Outputs:

```text
outputs/all_events.parquet
outputs/summary.csv
outputs/figures/*.png
```

## Current graphs

These figures are generated from the current local summary output and committed because they are small. Recreate them with the `plots` command below.

![Win rate by event type](outputs/figures/win_rate_by_event_type.png)

![Mean aligned return by event type](outputs/figures/mean_return_by_event_type.png)

![MSS aligned return distribution](outputs/figures/mss_aligned_return_distribution.png)

![Sample size by event type](outputs/figures/sample_size_by_event_type.png)

![Win rate by momentum bucket](outputs/figures/win_rate_by_momentum_bucket.png)

![Win rate by relative volume bucket](outputs/figures/win_rate_by_relative_volume_bucket.png)

## Create graphs from existing summary

```bash
PYTHONPATH=src python3 -m mss_research plots --summary outputs/summary.csv --out outputs/figures
```

Graphs include win rate, mean aligned return, MSS P25/mean/P75 aligned-return distribution, sample size, momentum buckets, and relative-volume buckets.

## Tests

```bash
python3 -m pytest -q
```
