from __future__ import annotations

from pathlib import Path

import polars as pl

from .features import (
    StudyConfig,
    add_divergences,
    add_indicators,
    detect_intermediate_swings,
    detect_mss_events,
    detect_swings,
    divergence_events,
    label_forward_returns,
    resample_ohlcv,
    summarize_events,
)


def instrument_from_path(path: Path) -> str:
    return path.stem.split("_")[0].upper()


def run_file(path: Path, out_dir: Path, config: StudyConfig = StudyConfig()) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = pl.read_parquet(path)
    instrument = instrument_from_path(path)
    all_events: list[pl.DataFrame] = []

    for timeframe in config.timeframes:
        bars = resample_ohlcv(raw, timeframe)
        bars = detect_swings(bars, k=config.swing_k)
        bars = detect_intermediate_swings(bars, k=config.swing_k)
        bars = add_indicators(bars, rsi_period=config.rsi_period, rolling_window=config.rolling_window)
        bars = add_divergences(bars)

        events = []
        for tier in ("short", "intermediate"):
            events.append(detect_mss_events(bars, tier=tier, k=config.swing_k))
        events.append(divergence_events(bars, "rsi"))
        events.append(divergence_events(bars, "volume"))
        events = [e for e in events if not e.is_empty()]
        if not events:
            continue
        frame = pl.concat(events, how="diagonal_relaxed")
        frame = frame.with_columns(pl.lit(instrument).alias("instrument"), pl.lit(timeframe).alias("timeframe"))
        frame = frame.select(["instrument", "timeframe", *[c for c in frame.columns if c not in {"instrument", "timeframe"}]])
        frame = label_forward_returns(frame, bars, config.horizons)
        all_events.append(frame)

    events_out = pl.concat(all_events, how="diagonal_relaxed") if all_events else pl.DataFrame()
    summary = summarize_events(events_out, config.horizons, config.bootstrap_iterations, config.random_seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    events_out.write_parquet(out_dir / f"{instrument.lower()}_events.parquet")
    summary.write_csv(out_dir / f"{instrument.lower()}_summary.csv")
    return events_out, summary


def run_directory(data_dir: Path, out_dir: Path, config: StudyConfig = StudyConfig()) -> tuple[pl.DataFrame, pl.DataFrame]:
    paths = sorted(data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files found in {data_dir}")
    events = []
    summaries = []
    for path in paths:
        e, s = run_file(path, out_dir, config)
        events.append(e)
        summaries.append(s)
    events_all = pl.concat(events, how="diagonal_relaxed") if events else pl.DataFrame()
    summary_all = pl.concat(summaries, how="diagonal_relaxed") if summaries else pl.DataFrame()
    events_all.write_parquet(out_dir / "all_events.parquet")
    summary_all.write_csv(out_dir / "summary.csv")
    return events_all, summary_all
