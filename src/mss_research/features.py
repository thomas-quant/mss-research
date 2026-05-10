from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


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
    last_swing_high_before = np.full(n, -1, dtype=int)
    last_swing_low_before = np.full(n, -1, dtype=int)
    last_high = -1
    last_low = -1
    short_high_flags = out["swing_high"].to_numpy(dtype=bool) if "swing_high" in out.columns else np.zeros(n, dtype=bool)
    short_low_flags = out["swing_low"].to_numpy(dtype=bool) if "swing_low" in out.columns else np.zeros(n, dtype=bool)
    for idx in range(n):
        last_swing_high_before[idx] = last_high
        last_swing_low_before[idx] = last_low
        if short_high_flags[idx]:
            last_high = idx
        if short_low_flags[idx]:
            last_low = idx

    high_schedule: dict[int, list[int]] = {}
    low_schedule: dict[int, list[int]] = {}
    high_flags = out[high_col].to_numpy(dtype=bool)
    low_flags = out[low_col].to_numpy(dtype=bool)
    high_avail = out[high_avail_col].to_numpy(dtype=float)
    low_avail = out[low_avail_col].to_numpy(dtype=float)

    for swing_i in np.flatnonzero(high_flags):
        avail = high_avail[swing_i]
        if np.isfinite(avail) and 0 <= int(avail) < n:
            high_schedule.setdefault(int(avail), []).append(int(swing_i))
    for swing_i in np.flatnonzero(low_flags):
        avail = low_avail[swing_i]
        if np.isfinite(avail) and 0 <= int(avail) < n:
            low_schedule.setdefault(int(avail), []).append(int(swing_i))

    events: list[dict] = []
    active_high_idx: int | None = None
    active_high_price = np.nan
    active_high_avail = -1
    active_low_idx: int | None = None
    active_low_price = np.nan
    active_low_avail = -1

    for i in range(n):
        for swing_i in high_schedule.get(i, []):
            active_high_idx = swing_i
            active_high_price = high[swing_i]
            active_high_avail = i
        for swing_i in low_schedule.get(i, []):
            active_low_idx = swing_i
            active_low_price = low[swing_i]
            active_low_avail = i

        if active_high_idx is not None and i > active_high_avail and high[i] > active_high_price:
            leg_start_idx = int(last_swing_low_before[i]) if last_swing_low_before[i] >= 0 else max(0, i - 1)
            events.append(_event_row(out, i, 1, tier, active_high_idx, active_high_price, close[i] > active_high_price, leg_start_idx))
            active_high_idx = None

        if active_low_idx is not None and i > active_low_avail and low[i] < active_low_price:
            leg_start_idx = int(last_swing_high_before[i]) if last_swing_high_before[i] >= 0 else max(0, i - 1)
            events.append(_event_row(out, i, -1, tier, active_low_idx, active_low_price, close[i] < active_low_price, leg_start_idx))
            active_low_idx = None

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
    """Context for the impulse leg that creates the MSS.

    Bullish MSS leg starts at latest short-term swing low before the break.
    Bearish MSS leg starts at latest short-term swing high before the break.
    RSI is direction-aligned: high RSI for bullish, low RSI inverted as 100-RSI for bearish.
    """
    leg_start_idx = int(max(0, min(leg_start_idx, event_idx)))
    leg = df.iloc[leg_start_idx : event_idx + 1]
    bar_count = int(len(leg))
    leg_volume_sum = float(leg["Volume"].sum())
    baseline_volume = df.iloc[event_idx].get("rolling_volume_median", np.nan)
    denom = float(baseline_volume) * bar_count if pd.notna(baseline_volume) and float(baseline_volume) > 0 else np.nan
    leg_relative_volume = leg_volume_sum / denom if pd.notna(denom) and denom > 0 else np.nan
    if direction == 1:
        leg_rsi_extreme = float(leg["rsi"].max()) if "rsi" in leg else np.nan
        leg_rsi_aligned = leg_rsi_extreme
    else:
        leg_rsi_extreme = float(leg["rsi"].min()) if "rsi" in leg else np.nan
        leg_rsi_aligned = 100.0 - leg_rsi_extreme if pd.notna(leg_rsi_extreme) else np.nan
    start_close = float(df.iloc[leg_start_idx]["Close"])
    end_close = float(df.iloc[event_idx]["Close"])
    leg_aligned_return = direction * (end_close / start_close - 1.0) if start_close else np.nan
    return {
        "leg_start_idx": leg_start_idx,
        "leg_bar_count": bar_count,
        "leg_volume_sum": leg_volume_sum,
        "leg_relative_volume": leg_relative_volume,
        "leg_volume_bucket": _bucket_value(leg_relative_volume, (0.8, 1.2), ("low", "normal", "high")),
        "leg_rsi_extreme": leg_rsi_extreme,
        "leg_rsi_aligned": leg_rsi_aligned,
        "leg_rsi_momentum_bucket": _bucket_value(leg_rsi_aligned, (55.0, 65.0), ("low", "medium", "high")),
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
            "swing_tier",
            "closed_through",
            "momentum_bucket",
            "relative_volume_bucket",
            "broken_swing_rsi_divergence",
            "broken_swing_volume_divergence",
            "leg_rsi_momentum_bucket",
            "leg_volume_bucket",
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
