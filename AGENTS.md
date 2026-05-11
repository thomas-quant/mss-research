# Agents Context: MSS Research

## Project purpose

This repo tests whether retail-trader market structure concepts predict future direction in ES/NQ intraday futures data. The current research focus is event studies around:

- Market structure shifts (MSS)
- Change in state of delivery (CISD)
- RSI divergences
- Volume divergences
- Context filters: structure prominence, break type, momentum, volume, time of day, and leg comparisons

The working conclusion so far: **MSS alone has not looked directionally predictive** under fixed forward-return labels. It may be more useful as context, volatility/regime marker, or with filters.

## Data and outputs policy

Local raw data exists under `data/` and is ignored by git:

- `data/es_1m.parquet`
- `data/nq_1m.parquet`

Required data columns:

- `datetime_utc`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

Large generated result tables under `outputs/` are ignored by git, especially:

- `outputs/all_events.parquet`
- `outputs/summary.csv`
- per-instrument event/summary files

Small matplotlib figures under `outputs/figures/*.png` are tracked and embedded in `README.md`.

## Current package and commands

Code lives in `src/mss_research/`. The core research pipeline uses Polars for data processing and parquet/CSV I/O; plot code may convert to pandas at the matplotlib boundary.

Main CLI:

```bash
PYTHONPATH=src python3 -m mss_research run --data data --out outputs --timeframes 15min --horizons 5,15,30,60 --bootstrap 100 --plots
```

Create plots from existing outputs:

```bash
PYTHONPATH=src python3 -m mss_research plots --summary outputs/summary.csv --events outputs/all_events.parquet --out outputs/figures
```

Tests:

```bash
python3 -m pytest -q
```

Latest verified test state when this file was written: `32 passed`.

## Market structure definitions

### Short-term swings

Short-term swing highs/lows use strict fractal `k=1`:

- Swing high: high greater than one bar on each side.
- Swing low: low lower than one bar on each side.
- Swings are only available after right-side confirmation.

### Intermediate swings

Intermediate swings are “swings of swings”:

- Intermediate swing high: a short-term swing high with lower short-term swing highs on both sides.
- Intermediate swing low: a short-term swing low with higher short-term swing lows on both sides.
- Intermediate swings are only available after the confirming swing on the right is available.

### Leg-based MSS

MSS was updated to be leg-based because the broken level must be left of the extremity that creates the shift.

Bullish MSS:

- Identify the lowest swing-low extremity of the displacement leg.
- The broken swing high must be left of that swing-low extremity.
- A swing high formed after the low does **not** qualify as the bullish shift level.
- Event fires when price trades through that eligible left-side swing high.

Bearish MSS mirrors bullish:

- Identify the highest swing-high extremity of the displacement leg.
- The broken swing low must be left of that swing-high extremity.
- A swing low formed after the high does **not** qualify as the bearish shift level.
- Event fires when price trades through that eligible left-side swing low.

MSS captures both:

- `traded_through`
- `closed_through`

### CISD standalone events

Change in state of delivery (CISD) is tracked as standalone event rows around confirmed short-term swing extrema.

Bullish CISD:

- Anchor = confirmed short-term swing low.
- Left run = at least 3 contiguous down-close candles into/ending at the swing-low area.
- `open` setup level = open of the top candle in the down-close run.
- `extreme` setup level = maximum high of the down-close run.
- Event fires only when a later available candle closes above the setup level.

Bearish CISD mirrors bullish:

- Anchor = confirmed short-term swing high.
- Left run = at least 3 contiguous up-close candles into/ending at the swing-high area.
- `open` setup level = open of the bottom candle in the up-close run.
- `extreme` setup level = minimum low of the up-close run.
- Event fires only when a later available candle closes below the setup level.

Tracked fields:

- `cisd_break_level_type`
- `cisd_anchor_idx`
- `cisd_run_start_idx`
- `cisd_run_end_idx`
- `cisd_run_length`
- `cisd_break_level`

## Current features

### Break-candle features

- `momentum_ratio`: break candle body divided by rolling median range.
- `momentum_bucket`: low / medium / high.
- `relative_volume`: break candle volume divided by rolling median volume.
- `relative_volume_bucket`: low / normal / high.

### MSS right-leg context

Right leg = leg that creates the MSS:

- Bullish: swing-low extremity → MSS break bar.
- Bearish: swing-high extremity → MSS break bar.

Tracked fields:

- `leg_start_idx`
- `leg_bar_count`
- `leg_volume_sum`
- `leg_relative_volume`
- `leg_volume_bucket`
- `leg_rsi_extreme`
- `leg_rsi_aligned`
- `leg_rsi_momentum_bucket`
- `right_leg_rsi_mean`
- `right_leg_rsi_mean_aligned`
- `right_leg_rsi_mean_bucket`
- `leg_aligned_return`

RSI alignment:

- Bullish: higher RSI = stronger.
- Bearish: `100 - RSI`, so higher aligned value = stronger bearish momentum.

### MSS left/right leg RSI comparison

Left leg = leg immediately before the MSS extremity:

- Bullish: prior swing high → swing-low extremity.
- Bearish: prior swing low → swing-high extremity.

Tracked fields:

- `left_leg_start_idx`
- `left_leg_bar_count`
- `left_leg_rsi_mean`
- `left_leg_rsi_mean_aligned`
- `leg_rsi_mean_delta`
- `leg_rsi_mean_delta_bucket`

Delta definition:

```text
leg_rsi_mean_delta = right_leg_rsi_mean_aligned - left_leg_rsi_mean_aligned
```

Buckets:

- `weakening`: `< -5`
- `neutral`: `-5 to +5`
- `strengthening`: `> +5`

Purpose: test whether MSS works better when the shift leg has stronger directional RSI momentum than the prior leg.

### MSS left/right leg volume comparison

Volume comparison mirrors the RSI delta pattern, but uses length-normalized relative volume for each leg.

Tracked fields:

- `left_leg_volume_sum`
- `left_leg_relative_volume`
- `leg_relative_volume_delta`
- `leg_relative_volume_delta_bucket`

Delta definition:

```text
leg_relative_volume_delta = leg_relative_volume - left_leg_relative_volume
```

Buckets:

- `contracting`: `< -0.2`
- `neutral`: `-0.2 to +0.2`
- `expanding`: `> +0.2`

Purpose: test whether MSS works better when the shift leg has expanding or contracting relative volume versus the prior leg.

### Divergences

RSI and volume divergences are measured at matched short-term swings:

- Bearish RSI divergence: price makes higher swing high, RSI makes lower value.
- Bullish RSI divergence: price makes lower swing low, RSI makes higher value.
- Volume divergence uses swing-bar volume and flags lower volume on the price extension.

### Time-of-day context

Events are bucketed using `America/New_York` clock time:

- `asia`: 18:00–00:00 ET
- `london`: 02:00–05:00 ET
- `ny_am`: 08:30–12:00 ET
- `ny_pm`: 13:30–16:00 ET
- `other`: all remaining times

Tracked field:

- `event_session`

## Labels and summaries

Forward-return labels currently use fixed bar horizons:

- `5`
- `15`
- `30`
- `60`

For each event and horizon:

- `fwd_return_{horizon}`
- `aligned_return_{horizon}`
- `win_{horizon}`

Aligned return means positive is favorable in the event direction.

Summary stats include:

- event count `n`
- win rate
- bootstrap confidence intervals for win rate and mean aligned return
- mean aligned return
- P25 aligned return
- median aligned return
- P75 aligned return

## Current plots

Tracked plots in `outputs/figures/`:

- `win_rate_by_event_type.png`
- `mean_return_by_event_type.png`
- `win_rate_by_timeframe_and_event_type.png`
- `mean_return_by_timeframe_and_event_type.png`
- `p75_return_by_timeframe_and_event_type.png`
- `mss_aligned_return_distribution.png`
- `cisd_aligned_return_distribution.png`
- `cisd_p75_return_by_timeframe.png`
- `sample_size_by_event_type.png`
- `win_rate_by_momentum_bucket.png`
- `win_rate_by_relative_volume_bucket.png`
- `win_rate_by_leg_rsi_momentum_bucket.png`
- `win_rate_by_leg_volume_bucket.png`
- `win_rate_by_right_leg_rsi_mean_bucket.png`
- `win_rate_by_leg_rsi_mean_delta_bucket.png`
- `win_rate_by_leg_relative_volume_delta_bucket.png`
- `win_rate_by_cisd_break_level_type.png`
- `win_rate_by_time_of_day_session.png`
- `win_rate_by_session_and_leg_volume.png`

## Research observations so far

- MSS alone has looked non-predictive: win rates near 50%, mean aligned returns near zero, and P25/P75 roughly straddling zero.
- P25/P75 distribution indicates MSS may mark activity/volatility rather than direction.
- Adding context is the current direction: session, volume, right-leg momentum, and right-vs-left momentum comparison.
- NaN buckets were misleading when non-MSS events lacked MSS leg context; bucket plots now drop NaN bucket rows.

## Implementation notes for future agents

- Use TDD for feature changes. Tests live in `tests/test_features.py` and `tests/test_plots.py`.
- For all-timeframe comparisons on this machine, run 1min/5min/15min separately and combine summaries; a single all-timeframe run can exceed memory.
- Use `--bootstrap 0` for fast vectorized Polars summaries when confidence intervals are not needed.
- Do not commit raw data or large output tables.
- It is OK to commit small PNG figures under `outputs/figures/`.
- If changing MSS rules, add targeted synthetic fixtures first. The most important rule is that the broken level must be left of the relevant extremity.
- If adding new summary grouping fields, add them to `summarize_events()` group columns and consider adding a plot.
- If adding plot outputs, update `README.md` to embed them.
