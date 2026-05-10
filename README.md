# MSS Research

Event-study tools for testing whether retail trading features predict near-term market direction:

- market structure shifts (MSS)
- RSI divergences
- volume divergences
- MSS break parameters: trade-through vs close-through, momentum, relative volume, short-term vs intermediate-term structure

Raw data and generated outputs are intentionally ignored by git.

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

## Create graphs from existing summary

```bash
PYTHONPATH=src python3 -m mss_research plots --summary outputs/summary.csv --out outputs/figures
```

Graphs include win rate, mean aligned return, sample size, momentum buckets, and relative-volume buckets.

## Tests

```bash
python3 -m pytest -q
```
