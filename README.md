# MSS Research

Polars-based event-study tools for testing whether retail trading features predict near-term market direction:

- market structure shifts (MSS)
- change in state of delivery (CISD)
- RSI divergences
- volume divergences
- MSS break parameters: trade-through vs close-through, momentum, relative volume, short-term vs intermediate-term structure
- MSS leg context: volume across the impulse leg and direction-aligned RSI momentum across that leg
- MSS leg momentum comparison: right-leg mean RSI, left-leg mean RSI, and right-minus-left momentum delta
- MSS leg volume comparison: right-leg relative volume, left-leg relative volume, and right-minus-left volume delta
- CISD standalone variants: close breaks through the run-start open and the run high/low extreme

Raw data and large generated output tables are intentionally ignored by git. Lightweight matplotlib figures in `outputs/figures/` are tracked so the README can show current research snapshots. The core research pipeline uses Polars; plotting still converts to pandas at the matplotlib boundary.

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

## Leg-based MSS definition

Bullish MSS only counts when price breaks a swing high that is left of the lowest swing-low extremity in the displacement leg. A swing high formed after that low is not eligible as the bullish shift level. Bearish MSS mirrors this: the broken swing low must be left of the highest swing-high extremity.

## CISD definition

Bullish CISD anchors on a confirmed short-term swing low with at least three contiguous down-close candles into the swing-low area. Bearish CISD mirrors this at a confirmed short-term swing high with at least three contiguous up-close candles. Up/down close means close-to-close direction.

Each setup emits two standalone CISD variants when a later confirmed bar closes through the setup level:

- `open`: bullish breaks above the open of the top candle in the down-close run; bearish breaks below the open of the bottom candle in the up-close run.
- `extreme`: bullish breaks above the maximum high of the run; bearish breaks below the minimum low of the run.

CISD event rows include `cisd_break_level_type`, `cisd_anchor_idx`, `cisd_run_start_idx`, `cisd_run_end_idx`, `cisd_run_length`, and `cisd_break_level`.

## Time-of-day sessions

Events are bucketed by America/New_York clock time:

- `asia`: 18:00-00:00 ET
- `london`: 02:00-05:00 ET
- `ny_am`: 08:30-12:00 ET
- `ny_pm`: 13:30-16:00 ET
- `other`: all remaining bars

## Run study

```bash
PYTHONPATH=src python3 -m mss_research run --data data --out outputs --plots
```

Useful faster run:

```bash
PYTHONPATH=src python3 -m mss_research run --data data --out outputs --timeframes 15min --horizons 5,15,30,60 --bootstrap 100 --plots
```

For all-timeframe comparisons on memory-limited machines, run each timeframe separately and combine the summary CSVs. Use `--bootstrap 0` for fast vectorized Polars summaries when confidence intervals are not needed.

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

![Win rate by timeframe and event type](outputs/figures/win_rate_by_timeframe_and_event_type.png)

![Mean aligned return by timeframe and event type](outputs/figures/mean_return_by_timeframe_and_event_type.png)

![P75 aligned return by timeframe and event type](outputs/figures/p75_return_by_timeframe_and_event_type.png)

![MSS aligned return distribution](outputs/figures/mss_aligned_return_distribution.png)

![CISD aligned return distribution](outputs/figures/cisd_aligned_return_distribution.png)

![CISD P75 aligned return by timeframe](outputs/figures/cisd_p75_return_by_timeframe.png)

![Sample size by event type](outputs/figures/sample_size_by_event_type.png)

![Win rate by momentum bucket](outputs/figures/win_rate_by_momentum_bucket.png)

![Win rate by MSS leg RSI momentum bucket](outputs/figures/win_rate_by_leg_rsi_momentum_bucket.png)

![Win rate by right-leg mean RSI bucket](outputs/figures/win_rate_by_right_leg_rsi_mean_bucket.png)

![Win rate by right-vs-left leg RSI momentum delta](outputs/figures/win_rate_by_leg_rsi_mean_delta_bucket.png)

![Win rate by right-vs-left leg relative volume delta](outputs/figures/win_rate_by_leg_relative_volume_delta_bucket.png)

![Win rate by CISD break level type](outputs/figures/win_rate_by_cisd_break_level_type.png)

![Win rate by MSS leg volume bucket](outputs/figures/win_rate_by_leg_volume_bucket.png)

![Win rate by relative volume bucket](outputs/figures/win_rate_by_relative_volume_bucket.png)

![Win rate by time-of-day session](outputs/figures/win_rate_by_time_of_day_session.png)

![Win rate by session and MSS leg volume](outputs/figures/win_rate_by_session_and_leg_volume.png)

## Create graphs from existing summary

```bash
PYTHONPATH=src python3 -m mss_research plots --summary outputs/summary.csv --out outputs/figures
```

Graphs include win rate, mean aligned return, timeframe-by-event comparisons, P75 aligned-return comparisons, MSS and CISD P25/mean/P75 aligned-return distributions, sample size, break-candle momentum/volume buckets, MSS-leg RSI/volume buckets, right-leg mean RSI, right-vs-left RSI delta, right-vs-left relative-volume delta, CISD break-level type, time-of-day sessions, and session-by-leg-volume heatmaps.

## Tests

```bash
python3 -m pytest -q
```
