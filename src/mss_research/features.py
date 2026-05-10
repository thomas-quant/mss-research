from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isnan
from typing import Iterable, Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
NY_TZ = ZoneInfo("America/New_York")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def assign_time_of_day_session(timestamp: Any) -> str:
    """Assign ICT-style session bucket using America/New_York clock time."""
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    if isinstance(timestamp, np.datetime64):
        timestamp = datetime.fromtimestamp(timestamp.astype("datetime64[ns]").astype("int") / 1_000_000_000, tz=timezone.utc)
    if not isinstance(timestamp, datetime):
        timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    local = timestamp.astimezone(NY_TZ)
    minutes = local.hour * 60 + local.minute
    if 18 * 60 <= minutes or minutes < 0:
        return "asia"
    if 2 * 60 <= minutes < 5 * 60:
        return "london"
    if 8 * 60 + 30 <= minutes < 12 * 60:
        return "ny_am"
    if 13 * 60 + 30 <= minutes < 16 * 60:
        return "ny_pm"
    return "other"


@dataclass(frozen=True)
class StudyConfig:
    timeframes: tuple[str, ...] = ("1min", "5min", "15min")
    horizons: tuple[int, ...] = (5, 15, 30, 60)
    swing_k: int = 1
    rsi_period: int = 14
    rolling_window: int = 50
    bootstrap_iterations: int = 1000
    random_seed: int = 7


def _to_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df.clone()
    return pl.from_pandas(df)


def _require_columns(df: pl.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _prep(df: Any) -> pl.DataFrame:
    out = _to_polars(df)
    _require_columns(out, ["datetime_utc", *OHLCV])
    return out.with_columns(pl.col("datetime_utc").cast(pl.Datetime(time_zone="UTC"))).sort("datetime_utc")


def _pl_timeframe(timeframe: str) -> str:
    return timeframe.replace("min", "m").replace("T", "m")


def resample_ohlcv(df: Any, timeframe: str) -> pl.DataFrame:
    out = _prep(df)
    if timeframe in {"1m", "1min", "1T"}:
        return out
    every = _pl_timeframe(timeframe)
    return (
        out.group_by_dynamic("datetime_utc", every=every, period=every, closed="left", label="left")
        .agg(
            pl.col("Open").first(),
            pl.col("High").max(),
            pl.col("Low").min(),
            pl.col("Close").last(),
            pl.col("Volume").sum(),
        )
        .drop_nulls(["Open", "High", "Low", "Close"])
        .sort("datetime_utc")
    )


def detect_swings(df: Any, k: int = 1) -> pl.DataFrame:
    if k < 1:
        raise ValueError("k must be >= 1")
    out = _to_polars(df)
    n = out.height
    high = out["High"].to_numpy().astype(float)
    low = out["Low"].to_numpy().astype(float)
    swing_high = np.ones(n, dtype=bool)
    swing_low = np.ones(n, dtype=bool)
    for offset in range(1, k + 1):
        high_prev = np.roll(high, offset); high_prev[:offset] = np.nan
        high_next = np.roll(high, -offset); high_next[-offset:] = np.nan
        low_prev = np.roll(low, offset); low_prev[:offset] = np.nan
        low_next = np.roll(low, -offset); low_next[-offset:] = np.nan
        swing_high &= (high > high_prev) & (high > high_next)
        swing_low &= (low < low_prev) & (low < low_next)
    idx = np.arange(n)
    high_avail = np.where(swing_high, idx + k, np.nan).astype(float)
    low_avail = np.where(swing_low, idx + k, np.nan).astype(float)
    high_avail[high_avail >= n] = np.nan
    low_avail[low_avail >= n] = np.nan
    return out.with_columns(
        pl.Series("swing_high", swing_high),
        pl.Series("swing_low", swing_low),
        pl.Series("swing_high_available_idx", high_avail),
        pl.Series("swing_low_available_idx", low_avail),
    )


def detect_intermediate_swings(df: Any, k: int = 1) -> pl.DataFrame:
    out = _to_polars(df)
    _require_columns(out, ["swing_high", "swing_low"])
    n = out.height
    intermediate_high = np.zeros(n, dtype=bool)
    intermediate_low = np.zeros(n, dtype=bool)
    high_avail = np.full(n, np.nan)
    low_avail = np.full(n, np.nan)
    high = out["High"].to_numpy()
    low = out["Low"].to_numpy()

    high_idx = np.flatnonzero(out["swing_high"].to_numpy().astype(bool))
    if len(high_idx) >= 3:
        prev_i, cur_i, next_i = high_idx[:-2], high_idx[1:-1], high_idx[2:]
        avail = next_i + k
        mask = (high[cur_i] > high[prev_i]) & (high[cur_i] > high[next_i]) & (avail < n)
        intermediate_high[cur_i[mask]] = True
        high_avail[cur_i[mask]] = avail[mask]

    low_idx = np.flatnonzero(out["swing_low"].to_numpy().astype(bool))
    if len(low_idx) >= 3:
        prev_i, cur_i, next_i = low_idx[:-2], low_idx[1:-1], low_idx[2:]
        avail = next_i + k
        mask = (low[cur_i] < low[prev_i]) & (low[cur_i] < low[next_i]) & (avail < n)
        intermediate_low[cur_i[mask]] = True
        low_avail[cur_i[mask]] = avail[mask]

    return out.with_columns(
        pl.Series("intermediate_swing_high", intermediate_high),
        pl.Series("intermediate_swing_low", intermediate_low),
        pl.Series("intermediate_swing_high_available_idx", high_avail),
        pl.Series("intermediate_swing_low_available_idx", low_avail),
    )


def compute_rsi(close: Any, period: int = 14) -> pl.Series:
    close_values = (close if isinstance(close, pl.Series) else pl.Series(close)).to_numpy().astype(float)
    delta = np.empty(len(close_values), dtype=float)
    delta[0] = np.nan
    delta[1:] = np.diff(close_values)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain[np.isnan(delta)] = np.nan
    loss[np.isnan(delta)] = np.nan
    avg_gain = _ewm_adjust_false_min_periods(gain, alpha=1 / period, min_periods=period)
    avg_loss = _ewm_adjust_false_min_periods(loss, alpha=1 / period, min_periods=period)
    rs = np.divide(avg_gain, avg_loss, out=np.full(len(avg_gain), np.nan), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50.0
    return pl.Series(rsi)


def _ewm_adjust_false_min_periods(values: np.ndarray, alpha: float, min_periods: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    avg = np.nan
    count = 0
    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        count += 1
        avg = value if np.isnan(avg) else ((1 - alpha) * avg + alpha * value)
        if count >= min_periods:
            out[i] = avg
    return out


def _bucket_array(values: np.ndarray, cuts: tuple[float, float], labels: tuple[str, str, str]) -> np.ndarray:
    # Match pandas.cut(..., bins=[-inf, cut0, cut1, inf]) default right-closed intervals.
    out = np.full(len(values), None, dtype=object)
    valid = ~np.isnan(values)
    out[valid & (values <= cuts[0])] = labels[0]
    out[valid & (values > cuts[0]) & (values <= cuts[1])] = labels[1]
    out[valid & (values > cuts[1])] = labels[2]
    return out


def add_indicators(df: Any, rsi_period: int = 14, rolling_window: int = 50) -> pl.DataFrame:
    out = _to_polars(df)
    rsi = compute_rsi(out["Close"], rsi_period)
    true_range = out["High"] - out["Low"]
    rolling_range = true_range.rolling_quantile(0.5, interpolation="linear", window_size=rolling_window, min_samples=1).shift(1)
    rolling_volume = out["Volume"].rolling_quantile(0.5, interpolation="linear", window_size=rolling_window, min_samples=1).shift(1)
    body = (out["Close"] - out["Open"]).abs()
    range_denom = rolling_range.replace(0, None)
    volume_denom = rolling_volume.replace(0, None)
    momentum_ratio = body / range_denom
    relative_volume = out["Volume"] / volume_denom
    return out.with_columns(
        rsi.alias("rsi"),
        rolling_range.alias("rolling_range_median"),
        rolling_volume.alias("rolling_volume_median"),
        body.alias("candle_body"),
        momentum_ratio.alias("momentum_ratio"),
        relative_volume.alias("relative_volume"),
        pl.Series("momentum_bucket", _bucket_array(momentum_ratio.to_numpy(), (0.5, 1.0), ("low", "medium", "high"))),
        pl.Series("relative_volume_bucket", _bucket_array(relative_volume.to_numpy(), (0.8, 1.2), ("low", "normal", "high"))),
    )


def add_divergences(df: Any, volume_measure: str = "swing_bar") -> pl.DataFrame:
    if volume_measure != "swing_bar":
        raise ValueError("only volume_measure='swing_bar' is implemented")
    out = _to_polars(df)
    _require_columns(out, ["swing_high", "swing_low", "rsi"])
    n = out.height
    rsi_div = np.zeros(n, dtype=int)
    vol_div = np.zeros(n, dtype=int)
    high = out["High"].to_numpy(); low = out["Low"].to_numpy(); rsi = out["rsi"].to_numpy(); volume = out["Volume"].to_numpy()
    high_idx = np.flatnonzero(out["swing_high"].to_numpy().astype(bool))
    if len(high_idx) >= 2:
        prev_i, cur_i = high_idx[:-1], high_idx[1:]
        price_extension = high[cur_i] > high[prev_i]
        rsi_div[cur_i[price_extension & (rsi[cur_i] < rsi[prev_i])]] = -1
        vol_div[cur_i[price_extension & (volume[cur_i] < volume[prev_i])]] = -1
    low_idx = np.flatnonzero(out["swing_low"].to_numpy().astype(bool))
    if len(low_idx) >= 2:
        prev_i, cur_i = low_idx[:-1], low_idx[1:]
        price_extension = low[cur_i] < low[prev_i]
        rsi_div[cur_i[price_extension & (rsi[cur_i] > rsi[prev_i])]] = 1
        vol_div[cur_i[price_extension & (volume[cur_i] < volume[prev_i])]] = 1
    return out.with_columns(pl.Series("rsi_divergence_direction", rsi_div), pl.Series("volume_divergence_direction", vol_div))


def _tier_columns(tier: str) -> tuple[str, str, str, str]:
    if tier == "short":
        return "swing_high", "swing_low", "swing_high_available_idx", "swing_low_available_idx"
    if tier == "intermediate":
        return "intermediate_swing_high", "intermediate_swing_low", "intermediate_swing_high_available_idx", "intermediate_swing_low_available_idx"
    raise ValueError("tier must be 'short' or 'intermediate'")


def _last_true_before(flags: np.ndarray) -> np.ndarray:
    result = np.full(len(flags), -1, dtype=int)
    last = -1
    for idx, flag in enumerate(flags):
        result[idx] = last
        if flag:
            last = idx
    return result


def detect_mss_events(df: Any, tier: str = "short", k: int = 1) -> pl.DataFrame:
    out = _to_polars(df)
    high_col, low_col, high_avail_col, low_avail_col = _tier_columns(tier)
    _require_columns(out, [high_col, low_col, high_avail_col, low_avail_col, "datetime_utc", *OHLCV])
    n = out.height
    high = out["High"].to_numpy(); low = out["Low"].to_numpy(); close = out["Close"].to_numpy()
    high_flags = out[high_col].to_numpy().astype(bool); low_flags = out[low_col].to_numpy().astype(bool)
    high_avail = out[high_avail_col].to_numpy(); low_avail = out[low_avail_col].to_numpy()
    last_swing_high_before = _last_true_before(high_flags)
    last_swing_low_before = _last_true_before(low_flags)

    short_high_flags = out["swing_high"].to_numpy().astype(bool) if "swing_high" in out.columns else high_flags
    short_low_flags = out["swing_low"].to_numpy().astype(bool) if "swing_low" in out.columns else low_flags
    ctx = _FrameContext(out, short_high_flags, short_low_flags)

    bullish_setup_schedule: dict[int, list[tuple[int, int]]] = {}
    bearish_setup_schedule: dict[int, list[tuple[int, int]]] = {}
    for low_idx in np.flatnonzero(low_flags):
        avail = low_avail[low_idx]
        broken_high_idx = int(last_swing_high_before[low_idx])
        if broken_high_idx >= 0 and np.isfinite(avail) and 0 <= int(avail) < n:
            bullish_setup_schedule.setdefault(int(avail), []).append((int(low_idx), broken_high_idx))
    for high_idx in np.flatnonzero(high_flags):
        avail = high_avail[high_idx]
        broken_low_idx = int(last_swing_low_before[high_idx])
        if broken_low_idx >= 0 and np.isfinite(avail) and 0 <= int(avail) < n:
            bearish_setup_schedule.setdefault(int(avail), []).append((int(high_idx), broken_low_idx))

    events: list[dict] = []
    active_bull_extremity_idx = active_high_idx = None
    active_high_price = np.nan; active_bull_avail = -1; active_bull_extremity_price = np.inf
    active_bear_extremity_idx = active_low_idx = None
    active_low_price = np.nan; active_bear_avail = -1; active_bear_extremity_price = -np.inf

    for i in range(n):
        for extremity_idx, broken_high_idx in bullish_setup_schedule.get(i, []):
            extremity_price = low[extremity_idx]
            if active_high_idx is None or extremity_price < active_bull_extremity_price:
                active_bull_extremity_idx = extremity_idx; active_bull_extremity_price = extremity_price
                active_high_idx = broken_high_idx; active_high_price = high[broken_high_idx]; active_bull_avail = i
        for extremity_idx, broken_low_idx in bearish_setup_schedule.get(i, []):
            extremity_price = high[extremity_idx]
            if active_low_idx is None or extremity_price > active_bear_extremity_price:
                active_bear_extremity_idx = extremity_idx; active_bear_extremity_price = extremity_price
                active_low_idx = broken_low_idx; active_low_price = low[broken_low_idx]; active_bear_avail = i

        if active_high_idx is not None and active_bull_extremity_idx is not None and i >= active_bull_avail and high[i] > active_high_price:
            events.append(_event_row(ctx, i, 1, tier, active_high_idx, active_high_price, close[i] > active_high_price, active_bull_extremity_idx))
            active_high_idx = None; active_bull_extremity_idx = None; active_bull_extremity_price = np.inf
        if active_low_idx is not None and active_bear_extremity_idx is not None and i >= active_bear_avail and low[i] < active_low_price:
            events.append(_event_row(ctx, i, -1, tier, active_low_idx, active_low_price, close[i] < active_low_price, active_bear_extremity_idx))
            active_low_idx = None; active_bear_extremity_idx = None; active_bear_extremity_price = -np.inf

    return pl.DataFrame(events) if events else pl.DataFrame()


def detect_cisd_events(df: Any, k: int = 1, min_run_length: int = 3) -> pl.DataFrame:
    """Detect standalone ICT change-in-state-of-delivery events at confirmed short swings.

    A bullish CISD anchors on a confirmed short swing low with at least ``min_run_length``
    contiguous down-close candles ending at the anchor. Bearish CISD mirrors this at
    swing highs with contiguous up-close candles. Both the run-start open level and
    the run extreme level are emitted as separate close-break event variants.
    """
    if min_run_length < 1:
        raise ValueError("min_run_length must be >= 1")
    out = _to_polars(df)
    _require_columns(out, ["swing_high", "swing_low", "swing_high_available_idx", "swing_low_available_idx", "datetime_utc", *OHLCV])
    n = out.height
    if n == 0:
        return pl.DataFrame()

    open_ = out["Open"].to_numpy().astype(float)
    high = out["High"].to_numpy().astype(float)
    low = out["Low"].to_numpy().astype(float)
    close = out["Close"].to_numpy().astype(float)
    swing_high = out["swing_high"].to_numpy().astype(bool)
    swing_low = out["swing_low"].to_numpy().astype(bool)
    high_avail = out["swing_high_available_idx"].to_numpy()
    low_avail = out["swing_low_available_idx"].to_numpy()
    down_close = np.zeros(n, dtype=bool)
    up_close = np.zeros(n, dtype=bool)
    down_close[1:] = close[1:] < close[:-1]
    up_close[1:] = close[1:] > close[:-1]

    ctx = _FrameContext(out, swing_high, swing_low)
    events: list[dict] = []

    for anchor_idx in np.flatnonzero(swing_low):
        avail = low_avail[anchor_idx]
        if not np.isfinite(avail) or int(avail) >= n:
            continue
        run_start, run_length = _directional_run_start(down_close, int(anchor_idx))
        if run_length < min_run_length:
            continue
        run_end = int(anchor_idx)
        levels = {
            "open": float(open_[run_start]),
            "extreme": float(np.max(high[run_start : run_end + 1])),
        }
        for level_type, level in levels.items():
            event_idx = _first_close_break(close, int(avail), level, direction=1)
            if event_idx is not None:
                events.append(_cisd_event_row(ctx, event_idx, 1, int(anchor_idx), run_start, run_end, run_length, level_type, level))

    for anchor_idx in np.flatnonzero(swing_high):
        avail = high_avail[anchor_idx]
        if not np.isfinite(avail) or int(avail) >= n:
            continue
        run_start, run_length = _directional_run_start(up_close, int(anchor_idx))
        if run_length < min_run_length:
            continue
        run_end = int(anchor_idx)
        levels = {
            "open": float(open_[run_start]),
            "extreme": float(np.min(low[run_start : run_end + 1])),
        }
        for level_type, level in levels.items():
            event_idx = _first_close_break(close, int(avail), level, direction=-1)
            if event_idx is not None:
                events.append(_cisd_event_row(ctx, event_idx, -1, int(anchor_idx), run_start, run_end, run_length, level_type, level))

    events.sort(key=lambda row: (row["event_idx"], row["direction"], row["cisd_break_level_type"], row["cisd_anchor_idx"]))
    return pl.DataFrame(events) if events else pl.DataFrame()


def _directional_run_start(direction_flags: np.ndarray, anchor_idx: int) -> tuple[int, int]:
    if anchor_idx <= 0 or not bool(direction_flags[anchor_idx]):
        return anchor_idx, 0
    run_start = int(anchor_idx)
    while run_start > 0 and bool(direction_flags[run_start]):
        run_start -= 1
    run_start += 1
    return run_start, int(anchor_idx - run_start + 1)


def _first_close_break(close: np.ndarray, start_idx: int, level: float, direction: int) -> int | None:
    if direction == 1:
        hits = np.flatnonzero(close[start_idx:] > level)
    else:
        hits = np.flatnonzero(close[start_idx:] < level)
    if len(hits) == 0:
        return None
    return int(start_idx + hits[0])


class _FrameContext:
    def __init__(self, df: pl.DataFrame, short_high_flags: np.ndarray, short_low_flags: np.ndarray):
        self.df = df
        self.columns = set(df.columns)
        self.datetime = df["datetime_utc"].to_list()
        self.high = df["High"].to_numpy(); self.low = df["Low"].to_numpy(); self.close = df["Close"].to_numpy(); self.volume = df["Volume"].to_numpy()
        self.volume_cumsum = np.concatenate([[0.0], self.volume.cumsum()])[:-1]
        self.rsi = df["rsi"].to_numpy() if "rsi" in self.columns else None
        self.rolling_volume = df["rolling_volume_median"].to_numpy() if "rolling_volume_median" in self.columns else np.full(df.height, np.nan)
        self.momentum_ratio = df["momentum_ratio"].to_numpy() if "momentum_ratio" in self.columns else np.full(df.height, np.nan)
        self.relative_volume = df["relative_volume"].to_numpy() if "relative_volume" in self.columns else np.full(df.height, np.nan)
        self.momentum_bucket = df["momentum_bucket"].to_list() if "momentum_bucket" in self.columns else [None] * df.height
        self.relative_volume_bucket = df["relative_volume_bucket"].to_list() if "relative_volume_bucket" in self.columns else [None] * df.height
        self.rsi_div = df["rsi_divergence_direction"].to_numpy() if "rsi_divergence_direction" in self.columns else np.zeros(df.height)
        self.vol_div = df["volume_divergence_direction"].to_numpy() if "volume_divergence_direction" in self.columns else np.zeros(df.height)
        self.last_swing_high_before = _last_true_before(short_high_flags)
        self.last_swing_low_before = _last_true_before(short_low_flags)


def _event_row(ctx: _FrameContext, i: int, direction: int, tier: str, swing_idx: int, swing_price: float, closed: bool, leg_start_idx: int) -> dict:
    leg = _leg_context(ctx, i, direction, leg_start_idx)
    return {
        "event_type": "mss",
        "event_idx": int(i),
        "datetime_utc": ctx.datetime[i],
        "event_session": assign_time_of_day_session(ctx.datetime[i]),
        "direction": int(direction),
        "swing_tier": tier,
        "broken_swing_idx": int(swing_idx),
        "broken_swing_price": float(swing_price),
        "traded_through": True,
        "closed_through": bool(closed),
        "momentum_ratio": float(ctx.momentum_ratio[i]) if not _is_missing(ctx.momentum_ratio[i]) else np.nan,
        "relative_volume": float(ctx.relative_volume[i]) if not _is_missing(ctx.relative_volume[i]) else np.nan,
        "momentum_bucket": ctx.momentum_bucket[i],
        "relative_volume_bucket": ctx.relative_volume_bucket[i],
        "broken_swing_rsi_divergence": bool(ctx.rsi_div[swing_idx] == -direction),
        "broken_swing_volume_divergence": bool(ctx.vol_div[swing_idx] == -direction),
        **leg,
    }


def _cisd_event_row(
    ctx: _FrameContext,
    i: int,
    direction: int,
    anchor_idx: int,
    run_start_idx: int,
    run_end_idx: int,
    run_length: int,
    level_type: str,
    level: float,
) -> dict:
    return {
        "event_type": "cisd",
        "event_idx": int(i),
        "datetime_utc": ctx.datetime[i],
        "event_session": assign_time_of_day_session(ctx.datetime[i]),
        "direction": int(direction),
        "swing_tier": "short",
        "traded_through": True,
        "closed_through": True,
        "momentum_ratio": float(ctx.momentum_ratio[i]) if not _is_missing(ctx.momentum_ratio[i]) else np.nan,
        "relative_volume": float(ctx.relative_volume[i]) if not _is_missing(ctx.relative_volume[i]) else np.nan,
        "momentum_bucket": ctx.momentum_bucket[i],
        "relative_volume_bucket": ctx.relative_volume_bucket[i],
        "cisd_break_level_type": level_type,
        "cisd_anchor_idx": int(anchor_idx),
        "cisd_run_start_idx": int(run_start_idx),
        "cisd_run_end_idx": int(run_end_idx),
        "cisd_run_length": int(run_length),
        "cisd_break_level": float(level),
    }


def _range_sum(cumsum_at_index: np.ndarray, values: np.ndarray, start_idx: int, end_idx: int) -> float:
    return float(cumsum_at_index[end_idx] + values[end_idx] - cumsum_at_index[start_idx])


def _leg_relative_volume(volume_sum: float, bar_count: int, baseline_volume: float) -> float:
    denom = float(baseline_volume) * bar_count if not _is_missing(baseline_volume) and float(baseline_volume) > 0 and bar_count > 0 else np.nan
    return volume_sum / denom if not _is_missing(denom) and denom > 0 else np.nan


def _bucket_value(value: float, cuts: tuple[float, float], labels: tuple[str, str, str]) -> str | float:
    if _is_missing(value):
        return np.nan
    if value < cuts[0]:
        return labels[0]
    if value < cuts[1]:
        return labels[1]
    return labels[2]


def _leg_context(ctx_or_df: Any, event_idx: int, direction: int, leg_start_idx: int) -> dict:
    ctx = ctx_or_df if isinstance(ctx_or_df, _FrameContext) else _FrameContext(_to_polars(ctx_or_df), _to_polars(ctx_or_df)["swing_high"].to_numpy().astype(bool), _to_polars(ctx_or_df)["swing_low"].to_numpy().astype(bool))
    leg_start_idx = int(max(0, min(leg_start_idx, event_idx)))
    right_bar_count = int(event_idx - leg_start_idx + 1)
    leg_volume_sum = _range_sum(ctx.volume_cumsum, ctx.volume, leg_start_idx, event_idx)
    baseline_volume = ctx.rolling_volume[event_idx]
    leg_relative_volume = _leg_relative_volume(leg_volume_sum, right_bar_count, baseline_volume)
    right_rsi = ctx.rsi[leg_start_idx : event_idx + 1] if ctx.rsi is not None else None

    if direction == 1:
        right_extreme = float(np.max(right_rsi)) if right_rsi is not None else np.nan
        right_extreme_aligned = right_extreme
        right_mean = float(np.mean(right_rsi)) if right_rsi is not None else np.nan
        right_mean_aligned = right_mean
        candidate = int(ctx.last_swing_high_before[leg_start_idx])
        left_start_idx = candidate if candidate >= 0 else np.nan
    else:
        right_extreme = float(np.min(right_rsi)) if right_rsi is not None else np.nan
        right_extreme_aligned = 100.0 - right_extreme if not _is_missing(right_extreme) else np.nan
        right_mean = float(np.mean(right_rsi)) if right_rsi is not None else np.nan
        right_mean_aligned = 100.0 - right_mean if not _is_missing(right_mean) else np.nan
        candidate = int(ctx.last_swing_low_before[leg_start_idx])
        left_start_idx = candidate if candidate >= 0 else np.nan

    if not _is_missing(left_start_idx):
        left_start_int = int(left_start_idx)
        left_rsi = ctx.rsi[left_start_int : leg_start_idx + 1] if ctx.rsi is not None else None
        left_mean = float(np.mean(left_rsi)) if left_rsi is not None else np.nan
        left_mean_aligned = 100.0 - left_mean if direction == 1 and not _is_missing(left_mean) else left_mean
        if direction == -1 and not _is_missing(left_mean):
            left_mean_aligned = left_mean
        left_bar_count = int(leg_start_idx - left_start_int + 1)
        left_volume_sum = _range_sum(ctx.volume_cumsum, ctx.volume, left_start_int, leg_start_idx)
        left_relative_volume = _leg_relative_volume(left_volume_sum, left_bar_count, baseline_volume)
    else:
        left_mean = np.nan; left_mean_aligned = np.nan; left_bar_count = 0; left_volume_sum = np.nan; left_relative_volume = np.nan

    leg_rsi_mean_delta = right_mean_aligned - left_mean_aligned if not _is_missing(right_mean_aligned) and not _is_missing(left_mean_aligned) else np.nan
    leg_relative_volume_delta = leg_relative_volume - left_relative_volume if not _is_missing(leg_relative_volume) and not _is_missing(left_relative_volume) else np.nan
    start_close = float(ctx.close[leg_start_idx]); end_close = float(ctx.close[event_idx])
    leg_aligned_return = direction * (end_close / start_close - 1.0) if start_close else np.nan
    return {
        "leg_start_idx": leg_start_idx,
        "leg_bar_count": right_bar_count,
        "leg_volume_sum": leg_volume_sum,
        "leg_relative_volume": leg_relative_volume,
        "leg_volume_bucket": _bucket_value(leg_relative_volume, (0.8, 1.2), ("low", "normal", "high")),
        "leg_rsi_extreme": right_extreme,
        "leg_rsi_aligned": right_extreme_aligned,
        "leg_rsi_momentum_bucket": _bucket_value(right_extreme_aligned, (55.0, 65.0), ("low", "medium", "high")),
        "right_leg_rsi_mean": right_mean,
        "right_leg_rsi_mean_aligned": right_mean_aligned,
        "right_leg_rsi_mean_bucket": _bucket_value(right_mean_aligned, (55.0, 65.0), ("low", "medium", "high")),
        "left_leg_start_idx": left_start_idx,
        "left_leg_bar_count": left_bar_count,
        "left_leg_volume_sum": left_volume_sum,
        "left_leg_relative_volume": left_relative_volume,
        "left_leg_rsi_mean": left_mean,
        "left_leg_rsi_mean_aligned": left_mean_aligned,
        "leg_rsi_mean_delta": leg_rsi_mean_delta,
        "leg_rsi_mean_delta_bucket": _bucket_value(leg_rsi_mean_delta, (-5.0, 5.0), ("weakening", "neutral", "strengthening")),
        "leg_relative_volume_delta": leg_relative_volume_delta,
        "leg_relative_volume_delta_bucket": _bucket_value(leg_relative_volume_delta, (-0.2, 0.2), ("contracting", "neutral", "expanding")),
        "leg_aligned_return": leg_aligned_return,
    }


def divergence_events(df: Any, divergence_type: str) -> pl.DataFrame:
    if divergence_type not in {"rsi", "volume"}:
        raise ValueError("divergence_type must be 'rsi' or 'volume'")
    out = _to_polars(df)
    col = f"{divergence_type}_divergence_direction"
    _require_columns(out, [col, "datetime_utc"])
    direction = out[col].to_numpy()
    idxs = np.flatnonzero(direction != 0)
    rows = []
    momentum_ratio = out["momentum_ratio"].to_numpy() if "momentum_ratio" in out.columns else np.full(out.height, np.nan)
    relative_volume = out["relative_volume"].to_numpy() if "relative_volume" in out.columns else np.full(out.height, np.nan)
    momentum_bucket = out["momentum_bucket"].to_list() if "momentum_bucket" in out.columns else [None] * out.height
    relative_volume_bucket = out["relative_volume_bucket"].to_list() if "relative_volume_bucket" in out.columns else [None] * out.height
    datetimes = out["datetime_utc"].to_list()
    for i in idxs:
        rows.append({
            "event_type": f"{divergence_type}_divergence",
            "event_idx": int(i),
            "datetime_utc": datetimes[i],
            "event_session": assign_time_of_day_session(datetimes[i]),
            "direction": int(direction[i]),
            "swing_tier": "short",
            "closed_through": None,
            "traded_through": None,
            "momentum_ratio": float(momentum_ratio[i]) if not _is_missing(momentum_ratio[i]) else np.nan,
            "relative_volume": float(relative_volume[i]) if not _is_missing(relative_volume[i]) else np.nan,
            "momentum_bucket": momentum_bucket[i],
            "relative_volume_bucket": relative_volume_bucket[i],
        })
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def label_forward_returns(events: Any, bars: Any, horizons: Iterable[int]) -> pl.DataFrame:
    out = _to_polars(events)
    bars_pl = _to_polars(bars)
    if out.is_empty():
        return out
    close = bars_pl["Close"].to_numpy()
    event_idx = out["event_idx"].to_numpy().astype(int)
    direction = out["direction"].to_numpy().astype(int)
    cols = []
    for horizon in horizons:
        target_i = event_idx + int(horizon)
        valid = target_i < len(close)
        fwd = np.full(out.height, np.nan)
        fwd[valid] = close[target_i[valid]] / close[event_idx[valid]] - 1.0
        aligned = fwd * direction
        wins = [bool(value > 0) if not np.isnan(value) else None for value in aligned]
        cols.extend([pl.Series(f"fwd_return_{horizon}", fwd), pl.Series(f"aligned_return_{horizon}", aligned), pl.Series(f"win_{horizon}", wins, dtype=pl.Boolean)])
    return out.with_columns(*cols)


def bootstrap_ci(values: Any, iterations: int = 1000, seed: int = 7, statistic: str = "mean") -> tuple[float, float]:
    samples = np.asarray(values.drop_nulls().to_numpy() if isinstance(values, pl.Series) else values, dtype=float)
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return np.nan, np.nan
    return _bootstrap_ci_from_samples(samples, iterations, seed, statistic)


def _bootstrap_ci_from_samples(samples: np.ndarray, iterations: int = 1000, seed: int = 7, statistic: str = "mean", index_cache: dict[int, np.ndarray] | None = None) -> tuple[float, float]:
    if len(samples) == 1 or np.all(samples == samples[0]):
        value = float(samples[0])
        return value, value
    if index_cache is not None:
        indices = index_cache.get(len(samples))
        if indices is None:
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(samples), size=(iterations, len(samples)))
            index_cache[len(samples)] = indices
        draws = samples[indices]
    else:
        rng = np.random.default_rng(seed)
        draws = rng.choice(samples, size=(iterations, len(samples)), replace=True)
    stats = draws.mean(axis=1) if statistic == "mean" else np.median(draws, axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_events(events: Any, horizons: Iterable[int], bootstrap_iterations: int = 1000, seed: int = 7) -> pl.DataFrame:
    events_pl = _to_polars(events)
    if events_pl.is_empty():
        return pl.DataFrame()
    group_cols = [
        c
        for c in [
            "instrument",
            "timeframe",
            "event_type",
            "event_session",
            "swing_tier",
            "closed_through",
            "momentum_bucket",
            "relative_volume_bucket",
            "broken_swing_rsi_divergence",
            "broken_swing_volume_divergence",
            "leg_rsi_momentum_bucket",
            "leg_volume_bucket",
            "right_leg_rsi_mean_bucket",
            "leg_rsi_mean_delta_bucket",
            "leg_relative_volume_delta_bucket",
            "cisd_break_level_type",
        ]
        if c in events_pl.columns
    ]
    rows = []
    index_cache: dict[int, np.ndarray] = {}
    for part in events_pl.partition_by(group_cols, maintain_order=True, include_key=True, as_dict=False):
        base = {c: part[c][0] for c in group_cols}
        for horizon in horizons:
            aligned = part[f"aligned_return_{horizon}"].drop_nulls().to_numpy()
            aligned = aligned[~np.isnan(aligned)]
            wins_raw = part[f"win_{horizon}"].drop_nulls().to_numpy()
            wins = np.asarray(wins_raw, dtype=bool).astype(float) if len(wins_raw) else np.array([], dtype=float)
            win_lo, win_hi = (np.nan, np.nan) if len(wins) == 0 else _bootstrap_ci_from_samples(wins, bootstrap_iterations, seed, "mean", index_cache)
            mean_lo, mean_hi = (np.nan, np.nan) if len(aligned) == 0 else _bootstrap_ci_from_samples(aligned.astype(float), bootstrap_iterations, seed, "mean", index_cache)
            rows.append({
                **base,
                "horizon": int(horizon),
                "n": int(len(aligned)),
                "win_rate": float(wins.mean()) if len(wins) else np.nan,
                "win_rate_ci_low": win_lo,
                "win_rate_ci_high": win_hi,
                "mean_aligned_return": float(aligned.mean()) if len(aligned) else np.nan,
                "mean_aligned_return_ci_low": mean_lo,
                "mean_aligned_return_ci_high": mean_hi,
                "p25_aligned_return": float(np.quantile(aligned, 0.25)) if len(aligned) else np.nan,
                "median_aligned_return": float(np.median(aligned)) if len(aligned) else np.nan,
                "p75_aligned_return": float(np.quantile(aligned, 0.75)) if len(aligned) else np.nan,
            })
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
