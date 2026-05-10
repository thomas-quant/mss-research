from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


NY_TZ = ZoneInfo("America/New_York")


def assign_time_of_day_session(timestamp: pd.Timestamp) -> str:
    """Assign ICT-style session bucket using America/New_York clock time.

    asia: 18:00-00:00 ET
    london: 02:00-05:00 ET
    ny_am: 08:30-12:00 ET
    ny_pm: 13:30-16:00 ET
    other: everything else
    """
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(NY_TZ)
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


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["datetime_utc", *OHLCV])
    out = df.copy()
    out["datetime_utc"] = pd.to_datetime(out["datetime_utc"], utc=True)
    out = out.sort_values("datetime_utc").reset_index(drop=True)
    return out


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    out = _prep(df)
    if timeframe in {"1m", "1min", "1T"}:
        return out
    resampled = (
        out.set_index("datetime_utc")
        .resample(timeframe, label="left", closed="left")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Open", "High", "Low", "Close"])
        .reset_index()
    )
    return resampled


def detect_swings(df: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    if k < 1:
        raise ValueError("k must be >= 1")
    out = df.copy().reset_index(drop=True)
    n = len(out)
    high = out["High"]
    low = out["Low"]
    swing_high = pd.Series(True, index=out.index)
    swing_low = pd.Series(True, index=out.index)
    for offset in range(1, k + 1):
        swing_high &= high.gt(high.shift(offset)) & high.gt(high.shift(-offset))
        swing_low &= low.lt(low.shift(offset)) & low.lt(low.shift(-offset))
    swing_high = swing_high.fillna(False)
    swing_low = swing_low.fillna(False)
    out["swing_high"] = swing_high.astype(bool)
    out["swing_low"] = swing_low.astype(bool)
    out["swing_high_available_idx"] = np.where(out["swing_high"], np.arange(n) + k, np.nan)
    out["swing_low_available_idx"] = np.where(out["swing_low"], np.arange(n) + k, np.nan)
    out.loc[out["swing_high_available_idx"] >= n, "swing_high_available_idx"] = np.nan
    out.loc[out["swing_low_available_idx"] >= n, "swing_low_available_idx"] = np.nan
    return out


def detect_intermediate_swings(df: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    _require_columns(df, ["swing_high", "swing_low"])
    out = df.copy().reset_index(drop=True)
    n = len(out)
    out["intermediate_swing_high"] = False
    out["intermediate_swing_low"] = False
    out["intermediate_swing_high_available_idx"] = np.nan
    out["intermediate_swing_low_available_idx"] = np.nan

    high_idx = out.index[out["swing_high"]].to_list()
    for prev_i, cur_i, next_i in zip(high_idx, high_idx[1:], high_idx[2:]):
        if out.at[cur_i, "High"] > out.at[prev_i, "High"] and out.at[cur_i, "High"] > out.at[next_i, "High"]:
            avail = next_i + k
            if avail < n:
                out.at[cur_i, "intermediate_swing_high"] = True
                out.at[cur_i, "intermediate_swing_high_available_idx"] = avail

    low_idx = out.index[out["swing_low"]].to_list()
    for prev_i, cur_i, next_i in zip(low_idx, low_idx[1:], low_idx[2:]):
        if out.at[cur_i, "Low"] < out.at[prev_i, "Low"] and out.at[cur_i, "Low"] < out.at[next_i, "Low"]:
            avail = next_i + k
            if avail < n:
                out.at[cur_i, "intermediate_swing_low"] = True
                out.at[cur_i, "intermediate_swing_low_available_idx"] = avail
    return out


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def add_indicators(df: pd.DataFrame, rsi_period: int = 14, rolling_window: int = 50) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["rsi"] = compute_rsi(out["Close"], rsi_period)
    true_range = out["High"] - out["Low"]
    out["rolling_range_median"] = true_range.rolling(rolling_window, min_periods=1).median().shift(1)
    out["rolling_volume_median"] = out["Volume"].rolling(rolling_window, min_periods=1).median().shift(1)
    out["candle_body"] = (out["Close"] - out["Open"]).abs()
    denom_range = out["rolling_range_median"].replace(0, np.nan)
    denom_volume = out["rolling_volume_median"].replace(0, np.nan)
    out["momentum_ratio"] = out["candle_body"] / denom_range
    out["relative_volume"] = out["Volume"] / denom_volume
    out["momentum_bucket"] = pd.cut(
        out["momentum_ratio"],
        bins=[-np.inf, 0.5, 1.0, np.inf],
        labels=["low", "medium", "high"],
    ).astype("object")
    out["relative_volume_bucket"] = pd.cut(
        out["relative_volume"],
        bins=[-np.inf, 0.8, 1.2, np.inf],
        labels=["low", "normal", "high"],
    ).astype("object")
    return out


def add_divergences(df: pd.DataFrame, volume_measure: str = "swing_bar") -> pd.DataFrame:
    if volume_measure != "swing_bar":
        raise ValueError("only volume_measure='swing_bar' is implemented")
    _require_columns(df, ["swing_high", "swing_low", "rsi"])
    out = df.copy().reset_index(drop=True)
    out["rsi_divergence_direction"] = 0
    out["volume_divergence_direction"] = 0

    high_idx = out.index[out["swing_high"]].to_list()
    for prev_i, cur_i in zip(high_idx, high_idx[1:]):
        price_extension = out.at[cur_i, "High"] > out.at[prev_i, "High"]
        if price_extension and out.at[cur_i, "rsi"] < out.at[prev_i, "rsi"]:
            out.at[cur_i, "rsi_divergence_direction"] = -1
        if price_extension and out.at[cur_i, "Volume"] < out.at[prev_i, "Volume"]:
            out.at[cur_i, "volume_divergence_direction"] = -1

    low_idx = out.index[out["swing_low"]].to_list()
    for prev_i, cur_i in zip(low_idx, low_idx[1:]):
        price_extension = out.at[cur_i, "Low"] < out.at[prev_i, "Low"]
        if price_extension and out.at[cur_i, "rsi"] > out.at[prev_i, "rsi"]:
            out.at[cur_i, "rsi_divergence_direction"] = 1
        if price_extension and out.at[cur_i, "Volume"] < out.at[prev_i, "Volume"]:
            out.at[cur_i, "volume_divergence_direction"] = 1
    return out


def _tier_columns(tier: str) -> tuple[str, str, str, str]:
    if tier == "short":
        return "swing_high", "swing_low", "swing_high_available_idx", "swing_low_available_idx"
    if tier == "intermediate":
        return (
            "intermediate_swing_high",
            "intermediate_swing_low",
            "intermediate_swing_high_available_idx",
            "intermediate_swing_low_available_idx",
        )
    raise ValueError("tier must be 'short' or 'intermediate'")


def detect_mss_events(df: pd.DataFrame, tier: str = "short", k: int = 1) -> pd.DataFrame:
    high_col, low_col, high_avail_col, low_avail_col = _tier_columns(tier)
    _require_columns(df, [high_col, low_col, high_avail_col, low_avail_col, "datetime_utc", *OHLCV])
    out = df.reset_index(drop=True)
    n = len(out)
    high = out["High"].to_numpy(dtype=float)
    low = out["Low"].to_numpy(dtype=float)
    close = out["Close"].to_numpy(dtype=float)
    high_flags = out[high_col].to_numpy(dtype=bool)
    low_flags = out[low_col].to_numpy(dtype=bool)
    high_avail = out[high_avail_col].to_numpy(dtype=float)
    low_avail = out[low_avail_col].to_numpy(dtype=float)

    last_swing_high_before = np.full(n, -1, dtype=int)
    last_swing_low_before = np.full(n, -1, dtype=int)
    last_high = -1
    last_low = -1
    for idx in range(n):
        last_swing_high_before[idx] = last_high
        last_swing_low_before[idx] = last_low
        if high_flags[idx]:
            last_high = idx
        if low_flags[idx]:
            last_low = idx

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
    active_bull_extremity_idx: int | None = None
    active_high_idx: int | None = None
    active_high_price = np.nan
    active_bull_avail = -1
    active_bull_extremity_price = np.inf
    active_bear_extremity_idx: int | None = None
    active_low_idx: int | None = None
    active_low_price = np.nan
    active_bear_avail = -1
    active_bear_extremity_price = -np.inf

    for i in range(n):
        for extremity_idx, broken_high_idx in bullish_setup_schedule.get(i, []):
            extremity_price = low[extremity_idx]
            if active_high_idx is None or extremity_price < active_bull_extremity_price:
                active_bull_extremity_idx = extremity_idx
                active_bull_extremity_price = extremity_price
                active_high_idx = broken_high_idx
                active_high_price = high[broken_high_idx]
                active_bull_avail = i
        for extremity_idx, broken_low_idx in bearish_setup_schedule.get(i, []):
            extremity_price = high[extremity_idx]
            if active_low_idx is None or extremity_price > active_bear_extremity_price:
                active_bear_extremity_idx = extremity_idx
                active_bear_extremity_price = extremity_price
                active_low_idx = broken_low_idx
                active_low_price = low[broken_low_idx]
                active_bear_avail = i

        if active_high_idx is not None and active_bull_extremity_idx is not None and i >= active_bull_avail and high[i] > active_high_price:
            events.append(
                _event_row(
                    out,
                    i,
                    1,
                    tier,
                    active_high_idx,
                    active_high_price,
                    close[i] > active_high_price,
                    active_bull_extremity_idx,
                )
            )
            active_high_idx = None
            active_bull_extremity_idx = None
            active_bull_extremity_price = np.inf

        if active_low_idx is not None and active_bear_extremity_idx is not None and i >= active_bear_avail and low[i] < active_low_price:
            events.append(
                _event_row(
                    out,
                    i,
                    -1,
                    tier,
                    active_low_idx,
                    active_low_price,
                    close[i] < active_low_price,
                    active_bear_extremity_idx,
                )
            )
            active_low_idx = None
            active_bear_extremity_idx = None
            active_bear_extremity_price = -np.inf

    result = pd.DataFrame(events)
    for col in ["traded_through", "closed_through", "broken_swing_rsi_divergence", "broken_swing_volume_divergence"]:
        if col in result:
            result[col] = result[col].astype(object)
    return result

def _event_row(
    df: pd.DataFrame,
    i: int,
    direction: int,
    tier: str,
    swing_idx: int,
    swing_price: float,
    closed: bool,
    leg_start_idx: int,
) -> dict:
    row = df.iloc[i]
    swing = df.iloc[swing_idx]
    leg = _leg_context(df, i, direction, leg_start_idx)
    return {
        "event_type": "mss",
        "event_idx": int(i),
        "datetime_utc": row["datetime_utc"],
        "event_session": assign_time_of_day_session(row["datetime_utc"]),
        "direction": int(direction),
        "swing_tier": tier,
        "broken_swing_idx": int(swing_idx),
        "broken_swing_price": float(swing_price),
        "traded_through": True,
        "closed_through": bool(closed),
        "momentum_ratio": row.get("momentum_ratio", np.nan),
        "relative_volume": row.get("relative_volume", np.nan),
        "momentum_bucket": row.get("momentum_bucket", np.nan),
        "relative_volume_bucket": row.get("relative_volume_bucket", np.nan),
        "broken_swing_rsi_divergence": bool(swing.get("rsi_divergence_direction", 0) == -direction),
        "broken_swing_volume_divergence": bool(swing.get("volume_divergence_direction", 0) == -direction),
        **leg,
    }


def _leg_context(df: pd.DataFrame, event_idx: int, direction: int, leg_start_idx: int) -> dict:
    """Context for right MSS leg and prior left leg.

    Right leg creates the MSS: extremity -> break bar.
    Left leg precedes the extremity: prior opposite swing -> extremity.
    RSI means are direction-aligned so higher always means stronger in that leg's direction.
    """
    leg_start_idx = int(max(0, min(leg_start_idx, event_idx)))
    right_leg = df.iloc[leg_start_idx : event_idx + 1]
    right_bar_count = int(len(right_leg))
    leg_volume_sum = float(right_leg["Volume"].sum())
    baseline_volume = df.iloc[event_idx].get("rolling_volume_median", np.nan)
    denom = float(baseline_volume) * right_bar_count if pd.notna(baseline_volume) and float(baseline_volume) > 0 else np.nan
    leg_relative_volume = leg_volume_sum / denom if pd.notna(denom) and denom > 0 else np.nan

    if direction == 1:
        right_extreme = float(right_leg["rsi"].max()) if "rsi" in right_leg else np.nan
        right_extreme_aligned = right_extreme
        right_mean = float(right_leg["rsi"].mean()) if "rsi" in right_leg else np.nan
        right_mean_aligned = right_mean
        left_candidates = df.index[(df["swing_high"].astype(bool)) & (df.index < leg_start_idx)].to_list()
        left_start_idx = int(left_candidates[-1]) if left_candidates else np.nan
    else:
        right_extreme = float(right_leg["rsi"].min()) if "rsi" in right_leg else np.nan
        right_extreme_aligned = 100.0 - right_extreme if pd.notna(right_extreme) else np.nan
        right_mean = float(right_leg["rsi"].mean()) if "rsi" in right_leg else np.nan
        right_mean_aligned = 100.0 - right_mean if pd.notna(right_mean) else np.nan
        left_candidates = df.index[(df["swing_low"].astype(bool)) & (df.index < leg_start_idx)].to_list()
        left_start_idx = int(left_candidates[-1]) if left_candidates else np.nan

    if pd.notna(left_start_idx):
        left_leg = df.iloc[int(left_start_idx) : leg_start_idx + 1]
        left_mean = float(left_leg["rsi"].mean()) if "rsi" in left_leg else np.nan
        # Prior leg direction is opposite the MSS shift direction.
        left_mean_aligned = 100.0 - left_mean if direction == 1 and pd.notna(left_mean) else left_mean
        if direction == -1 and pd.notna(left_mean):
            left_mean_aligned = left_mean
        left_bar_count = int(len(left_leg))
    else:
        left_mean = np.nan
        left_mean_aligned = np.nan
        left_bar_count = 0

    leg_rsi_mean_delta = right_mean_aligned - left_mean_aligned if pd.notna(right_mean_aligned) and pd.notna(left_mean_aligned) else np.nan
    start_close = float(df.iloc[leg_start_idx]["Close"])
    end_close = float(df.iloc[event_idx]["Close"])
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
        "left_leg_rsi_mean": left_mean,
        "left_leg_rsi_mean_aligned": left_mean_aligned,
        "leg_rsi_mean_delta": leg_rsi_mean_delta,
        "leg_rsi_mean_delta_bucket": _bucket_value(leg_rsi_mean_delta, (-5.0, 5.0), ("weakening", "neutral", "strengthening")),
        "leg_aligned_return": leg_aligned_return,
    }

def _bucket_value(value: float, cuts: tuple[float, float], labels: tuple[str, str, str]) -> str | float:
    if pd.isna(value):
        return np.nan
    if value < cuts[0]:
        return labels[0]
    if value < cuts[1]:
        return labels[1]
    return labels[2]


def divergence_events(df: pd.DataFrame, divergence_type: str) -> pd.DataFrame:
    if divergence_type not in {"rsi", "volume"}:
        raise ValueError("divergence_type must be 'rsi' or 'volume'")
    col = f"{divergence_type}_divergence_direction"
    _require_columns(df, [col, "datetime_utc"])
    rows = []
    for i, row in df[df[col] != 0].iterrows():
        rows.append(
            {
                "event_type": f"{divergence_type}_divergence",
                "event_idx": int(i),
                "datetime_utc": row["datetime_utc"],
                "event_session": assign_time_of_day_session(row["datetime_utc"]),
                "direction": int(row[col]),
                "swing_tier": "short",
                "closed_through": np.nan,
                "traded_through": np.nan,
                "momentum_ratio": row.get("momentum_ratio", np.nan),
                "relative_volume": row.get("relative_volume", np.nan),
                "momentum_bucket": row.get("momentum_bucket", np.nan),
                "relative_volume_bucket": row.get("relative_volume_bucket", np.nan),
            }
        )
    return pd.DataFrame(rows)


def label_forward_returns(events: pd.DataFrame, bars: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    out = events.copy().reset_index(drop=True)
    close = bars["Close"].reset_index(drop=True)
    for horizon in horizons:
        fwd_returns = []
        aligned_returns = []
        wins = []
        for _, event in out.iterrows():
            i = int(event["event_idx"])
            target_i = i + int(horizon)
            if target_i >= len(close):
                fwd = np.nan
                aligned = np.nan
                win = np.nan
            else:
                fwd = float(close.iloc[target_i] / close.iloc[i] - 1.0)
                aligned = fwd * int(event["direction"])
                win = bool(aligned > 0)
            fwd_returns.append(fwd)
            aligned_returns.append(aligned)
            wins.append(win)
        out[f"fwd_return_{horizon}"] = fwd_returns
        out[f"aligned_return_{horizon}"] = aligned_returns
        out[f"win_{horizon}"] = pd.Series(wins, dtype=object)
    return out


def bootstrap_ci(values: pd.Series, iterations: int = 1000, seed: int = 7, statistic: str = "mean") -> tuple[float, float]:
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = clean.to_numpy()
    stats = []
    for _ in range(iterations):
        draw = rng.choice(samples, size=len(samples), replace=True)
        stats.append(float(np.mean(draw) if statistic == "mean" else np.median(draw)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_events(events: pd.DataFrame, horizons: Iterable[int], bootstrap_iterations: int = 1000, seed: int = 7) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
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
        ]
        if c in events.columns
    ]
    rows = []
    for keys, group in events.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for horizon in horizons:
            aligned = group[f"aligned_return_{horizon}"].dropna()
            wins = group[f"win_{horizon}"].dropna().astype(bool)
            win_lo, win_hi = bootstrap_ci(wins.astype(float), bootstrap_iterations, seed, "mean")
            mean_lo, mean_hi = bootstrap_ci(aligned, bootstrap_iterations, seed, "mean")
            rows.append(
                {
                    **base,
                    "horizon": int(horizon),
                    "n": int(len(aligned)),
                    "win_rate": float(wins.mean()) if len(wins) else np.nan,
                    "win_rate_ci_low": win_lo,
                    "win_rate_ci_high": win_hi,
                    "mean_aligned_return": float(aligned.mean()) if len(aligned) else np.nan,
                    "mean_aligned_return_ci_low": mean_lo,
                    "mean_aligned_return_ci_high": mean_hi,
                    "p25_aligned_return": float(aligned.quantile(0.25)) if len(aligned) else np.nan,
                    "median_aligned_return": float(aligned.median()) if len(aligned) else np.nan,
                    "p75_aligned_return": float(aligned.quantile(0.75)) if len(aligned) else np.nan,
                }
            )
    return pd.DataFrame(rows)
