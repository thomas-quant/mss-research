from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def run_file(path: Path, out_dir: Path, config: StudyConfig = StudyConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(path)
    instrument = instrument_from_path(path)
    all_events = []

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
        events = [e for e in events if not e.empty]
        if not events:
            continue
        frame = pd.concat(events, ignore_index=True, sort=False)
        frame.insert(0, "instrument", instrument)
        frame.insert(1, "timeframe", timeframe)
        frame = label_forward_returns(frame, bars, config.horizons)
        all_events.append(frame)

    if all_events:
        events_out = pd.concat(all_events, ignore_index=True, sort=False)
    else:
        events_out = pd.DataFrame()
    summary = summarize_events(events_out, config.horizons, config.bootstrap_iterations, config.random_seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / f"{instrument.lower()}_events.parquet"
    summary_path = out_dir / f"{instrument.lower()}_summary.csv"
    events_out.to_parquet(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    return events_out, summary


def run_directory(data_dir: Path, out_dir: Path, config: StudyConfig = StudyConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files found in {data_dir}")
    events = []
    summaries = []
    for path in paths:
        e, s = run_file(path, out_dir, config)
        events.append(e)
        summaries.append(s)
    events_all = pd.concat(events, ignore_index=True, sort=False) if events else pd.DataFrame()
    summary_all = pd.concat(summaries, ignore_index=True, sort=False) if summaries else pd.DataFrame()
    events_all.to_parquet(out_dir / "all_events.parquet", index=False)
    summary_all.to_csv(out_dir / "summary.csv", index=False)
    return events_all, summary_all
