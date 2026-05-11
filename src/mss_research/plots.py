from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl


EVENT_CORRELATION_TYPES = ("cisd", "mss", "rsi_divergence", "volume_divergence")
EVENT_CORRELATION_LABELS = {
    "cisd": "CISD",
    "mss": "MSS",
    "rsi_divergence": "RSI div",
    "volume_divergence": "Vol div",
}


def _as_pandas(df):
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def compute_event_type_correlations(events: pl.DataFrame | pd.DataFrame, n_bars: int, timeframe: str) -> pl.DataFrame:
    """Compute same-bar binary Pearson/phi correlations between event types."""
    events_pl = events if isinstance(events, pl.DataFrame) else pl.from_pandas(events)
    required = {"instrument", "event_idx", "event_type"}
    missing = required - set(events_pl.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")

    base = (
        events_pl.select("instrument", "event_idx", "event_type")
        .filter(pl.col("event_type").is_in(EVENT_CORRELATION_TYPES))
        .unique()
    )
    sets = {
        event_type: base.filter(pl.col("event_type") == event_type).select("instrument", "event_idx").unique()
        for event_type in EVENT_CORRELATION_TYPES
    }
    counts = {event_type: sets[event_type].height for event_type in EVENT_CORRELATION_TYPES}
    rows = []
    for pos, event_a in enumerate(EVENT_CORRELATION_TYPES):
        for event_b in EVENT_CORRELATION_TYPES[pos + 1 :]:
            count_a = counts[event_a]
            count_b = counts[event_b]
            overlap = sets[event_a].join(sets[event_b], on=["instrument", "event_idx"], how="inner").height
            denom = math.sqrt(count_a * count_b * (n_bars - count_a) * (n_bars - count_b))
            phi = ((overlap * n_bars) - (count_a * count_b)) / denom if denom else float("nan")
            rows.append(
                {
                    "timeframe": timeframe,
                    "event_a": event_a,
                    "event_b": event_b,
                    "event_a_label": EVENT_CORRELATION_LABELS[event_a],
                    "event_b_label": EVENT_CORRELATION_LABELS[event_b],
                    "count_a": count_a,
                    "count_b": count_b,
                    "n_bars": int(n_bars),
                    "same_bar_overlap": int(overlap),
                    "pearson_phi": float(phi),
                    "pct_a_with_b": overlap / count_a if count_a else float("nan"),
                    "pct_b_with_a": overlap / count_b if count_b else float("nan"),
                }
            )
    return pl.DataFrame(rows)


def create_event_correlation_plot(correlations: pl.DataFrame | pd.DataFrame, out_dir: Path | str) -> Path:
    """Create Matplotlib heatmaps for same-bar event-type Pearson/phi correlation."""
    corr = _as_pandas(correlations)
    required = {"timeframe", "event_a", "event_b", "pearson_phi"}
    missing = required - set(corr.columns)
    if missing:
        raise ValueError(f"correlations missing columns: {sorted(missing)}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / "event_type_correlation_by_timeframe.png"

    timeframes = list(corr["timeframe"].dropna().unique())
    preferred = ["1min", "5min", "15min"]
    timeframes = [tf for tf in preferred if tf in timeframes] + [tf for tf in timeframes if tf not in preferred]
    labels = [EVENT_CORRELATION_LABELS[event_type] for event_type in EVENT_CORRELATION_TYPES]
    fig, axes = plt.subplots(1, len(timeframes), figsize=(5 * len(timeframes), 5), squeeze=False, constrained_layout=True)
    for ax, timeframe in zip(axes[0], timeframes):
        matrix = pd.DataFrame(0.0, index=EVENT_CORRELATION_TYPES, columns=EVENT_CORRELATION_TYPES)
        frame = corr[corr["timeframe"] == timeframe]
        for _, row in frame.iterrows():
            matrix.loc[row["event_a"], row["event_b"]] = float(row["pearson_phi"])
            matrix.loc[row["event_b"], row["event_a"]] = float(row["pearson_phi"])
        image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-0.35, vmax=0.35, cmap="RdBu_r")
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(f"{timeframe} same-bar Pearson/phi")
        for y in range(len(labels)):
            for x in range(len(labels)):
                value = matrix.iloc[y, x]
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Correlation")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _clean_summary(summary: pd.DataFrame) -> pd.DataFrame:
    summary = _as_pandas(summary)
    required = {"event_type", "horizon", "n", "win_rate", "mean_aligned_return"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary missing columns: {sorted(missing)}")
    out = summary.copy()
    out = out[out["n"].fillna(0).astype(float) > 0]
    out["event_label"] = out.apply(_event_label, axis=1)
    return out


def _event_label(row: pd.Series) -> str:
    event = str(row.get("event_type", "event"))
    tier = row.get("swing_tier")
    closed = row.get("closed_through")
    if event == "mss" and pd.notna(tier):
        suffix = "close" if closed is True or str(closed) == "True" else "trade"
        return f"mss/{tier}/{suffix}"
    return event


def _weighted_by_event(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for (event_label, horizon), group in summary.groupby(["event_label", "horizon"], dropna=False):
        weights = group["n"].astype(float)
        values = group[value_col].astype(float)
        if weights.sum() <= 0:
            continue
        rows.append(
            {
                "event_label": event_label,
                "horizon": int(horizon),
                value_col: float((values * weights).sum() / weights.sum()),
                "n": int(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def create_summary_plots(summary: pd.DataFrame, out_dir: Path | str) -> list[Path]:
    """Create compact PNG charts from summary.csv-style output."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data = _clean_summary(summary)
    if data.empty:
        return []

    paths: list[Path] = []
    paths.append(_line_plot(_weighted_by_event(data, "win_rate"), "win_rate", "Win rate", out_path / "win_rate_by_event_type.png", baseline=0.5))
    paths.append(
        _line_plot(
            _weighted_by_event(data, "mean_aligned_return"),
            "mean_aligned_return",
            "Mean aligned return",
            out_path / "mean_return_by_event_type.png",
            baseline=0.0,
        )
    )
    paths.append(_bar_plot(data, out_path / "sample_size_by_event_type.png"))
    if "timeframe" in data.columns:
        paths.append(
            _timeframe_event_plot(
                data,
                "win_rate",
                "Win rate",
                out_path / "win_rate_by_timeframe_and_event_type.png",
                baseline=0.5,
            )
        )
        paths.append(
            _timeframe_event_plot(
                data,
                "mean_aligned_return",
                "Mean aligned return",
                out_path / "mean_return_by_timeframe_and_event_type.png",
                baseline=0.0,
            )
        )
        if "p75_aligned_return" in data.columns:
            paths.append(
                _timeframe_event_plot(
                    data,
                    "p75_aligned_return",
                    "P75 aligned return",
                    out_path / "p75_return_by_timeframe_and_event_type.png",
                    baseline=0.0,
                )
            )
            if "cisd_break_level_type" in data.columns:
                paths.append(_cisd_timeframe_p75_plot(data, out_path / "cisd_p75_return_by_timeframe.png"))

    if "momentum_bucket" in data.columns:
        paths.append(_bucket_plot(data, "momentum_bucket", "win_rate", out_path / "win_rate_by_momentum_bucket.png"))
    if "relative_volume_bucket" in data.columns:
        paths.append(_bucket_plot(data, "relative_volume_bucket", "win_rate", out_path / "win_rate_by_relative_volume_bucket.png"))
    if "leg_rsi_momentum_bucket" in data.columns:
        paths.append(_bucket_plot(data, "leg_rsi_momentum_bucket", "win_rate", out_path / "win_rate_by_leg_rsi_momentum_bucket.png"))
    if "leg_volume_bucket" in data.columns:
        paths.append(_bucket_plot(data, "leg_volume_bucket", "win_rate", out_path / "win_rate_by_leg_volume_bucket.png"))
    if "right_leg_rsi_mean_bucket" in data.columns:
        paths.append(_bucket_plot(data, "right_leg_rsi_mean_bucket", "win_rate", out_path / "win_rate_by_right_leg_rsi_mean_bucket.png"))
    if "leg_rsi_mean_delta_bucket" in data.columns:
        paths.append(_bucket_plot(data, "leg_rsi_mean_delta_bucket", "win_rate", out_path / "win_rate_by_leg_rsi_mean_delta_bucket.png"))
    if "leg_relative_volume_delta_bucket" in data.columns:
        paths.append(_bucket_plot(data, "leg_relative_volume_delta_bucket", "win_rate", out_path / "win_rate_by_leg_relative_volume_delta_bucket.png"))
    if "cisd_break_level_type" in data.columns:
        paths.append(_bucket_plot(data, "cisd_break_level_type", "win_rate", out_path / "win_rate_by_cisd_break_level_type.png"))
    if "event_session" in data.columns:
        paths.append(_bucket_plot(data, "event_session", "win_rate", out_path / "win_rate_by_time_of_day_session.png"))
    if "event_session" in data.columns and "leg_volume_bucket" in data.columns:
        paths.append(_session_volume_heatmap(data, out_path / "win_rate_by_session_and_leg_volume.png"))

    return [p for p in paths if p.exists()]


def _line_plot(data: pd.DataFrame, y_col: str, ylabel: str, path: Path, baseline: float) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, group in data.groupby("event_label"):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], group[y_col], marker="o", label=str(label))
    ax.axhline(baseline, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Forward horizon (bars)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by event type")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _bar_plot(data: pd.DataFrame, path: Path) -> Path:
    counts = data.groupby("event_label")["n"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_xlabel("Event type")
    ax.set_ylabel("Summary sample count")
    ax.set_title("Sample size by event type")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _weighted_by_timeframe_event(summary: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for (timeframe, event_label, horizon), group in summary.groupby(["timeframe", "event_label", "horizon"], dropna=False):
        weights = group["n"].astype(float)
        values = group[value_col].astype(float)
        if weights.sum() <= 0:
            continue
        rows.append(
            {
                "timeframe": str(timeframe),
                "event_label": str(event_label),
                "plot_label": f"{timeframe}/{event_label}",
                "horizon": int(horizon),
                value_col: float((values * weights).sum() / weights.sum()),
                "n": int(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def _timeframe_event_plot(data: pd.DataFrame, y_col: str, ylabel: str, path: Path, baseline: float) -> Path:
    plot_df = _weighted_by_timeframe_event(data, y_col)
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, group in plot_df.groupby("plot_label"):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], group[y_col], marker="o", label=str(label))
    ax.axhline(baseline, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Forward horizon (bars)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by timeframe and event type")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _cisd_timeframe_p75_plot(data: pd.DataFrame, path: Path) -> Path:
    filtered = data[data["event_type"] == "cisd"].dropna(subset=["timeframe", "cisd_break_level_type", "p75_aligned_return"])
    rows = []
    for (timeframe, level_type, horizon), group in filtered.groupby(["timeframe", "cisd_break_level_type", "horizon"], dropna=False):
        weights = group["n"].astype(float)
        if weights.sum() <= 0:
            continue
        values = group["p75_aligned_return"].astype(float)
        rows.append(
            {
                "plot_label": f"{timeframe}/{level_type}",
                "horizon": int(horizon),
                "p75_aligned_return": float((values * weights).sum() / weights.sum()),
            }
        )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 7))
    for label, group in plot_df.groupby("plot_label"):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], group["p75_aligned_return"], marker="o", label=str(label))
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Forward horizon (bars)")
    ax.set_ylabel("P75 aligned return")
    ax.set_title("CISD P75 aligned return by timeframe and break-level type")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _bucket_plot(data: pd.DataFrame, bucket_col: str, y_col: str, path: Path) -> Path:
    rows = []
    data = data.dropna(subset=[bucket_col])
    for (bucket, horizon), group in data.groupby([bucket_col, "horizon"], dropna=False):
        weights = group["n"].astype(float)
        if weights.sum() <= 0:
            continue
        values = group[y_col].astype(float)
        rows.append({bucket_col: str(bucket), "horizon": int(horizon), y_col: float((values * weights).sum() / weights.sum())})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    for bucket, group in plot_df.groupby(bucket_col):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], group[y_col], marker="o", label=str(bucket))
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Forward horizon (bars)")
    ax.set_ylabel("Win rate")
    ax.set_title(f"Win rate by {bucket_col.replace('_', ' ')}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _session_volume_heatmap(data: pd.DataFrame, path: Path) -> Path:
    filtered = data[(data["event_type"] == "mss") if "event_type" in data.columns else True].dropna(subset=["event_session", "leg_volume_bucket"])
    rows = []
    for (session, volume_bucket), group in filtered.groupby(["event_session", "leg_volume_bucket"], dropna=False):
        weights = group["n"].astype(float)
        if weights.sum() <= 0:
            continue
        values = group["win_rate"].astype(float)
        rows.append({"event_session": str(session), "leg_volume_bucket": str(volume_bucket), "win_rate": float((values * weights).sum() / weights.sum())})
    heat = pd.DataFrame(rows)
    if heat.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No MSS session/volume data", ha="center", va="center")
    else:
        session_order = [s for s in ["asia", "london", "ny_am", "ny_pm", "other"] if s in set(heat["event_session"])]
        volume_order = [v for v in ["low", "normal", "high"] if v in set(heat["leg_volume_bucket"])]
        pivot = heat.pivot(index="event_session", columns="leg_volume_bucket", values="win_rate").reindex(index=session_order, columns=volume_order)
        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(pivot.to_numpy(dtype=float), vmin=0.4, vmax=0.6, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title("MSS win rate by session and leg-volume bucket")
        for y in range(len(pivot.index)):
            for x in range(len(pivot.columns)):
                val = pivot.iloc[y, x]
                if pd.notna(val):
                    ax.text(x, y, f"{val:.1%}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, label="Win rate")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def create_mss_distribution_plot(events: pd.DataFrame, out_dir: Path | str, horizons: list[int] | tuple[int, ...] | None = None) -> Path:
    """Plot MSS aligned-return P25/mean/P75 by horizon and structure subtype."""
    events = _as_pandas(events)
    required = {"event_type", "swing_tier", "closed_through"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    horizon_cols = [c for c in events.columns if c.startswith("aligned_return_")]
    if horizons is not None:
        wanted = {f"aligned_return_{int(h)}" for h in horizons}
        horizon_cols = [c for c in horizon_cols if c in wanted]
    if not horizon_cols:
        raise ValueError("events has no aligned_return_* columns")

    mss = events[events["event_type"] == "mss"].copy()
    if mss.empty:
        raise ValueError("events has no MSS rows")
    mss["event_label"] = mss.apply(_event_label, axis=1)

    rows = []
    for col in horizon_cols:
        horizon = int(col.rsplit("_", 1)[1])
        for label, group in mss.groupby("event_label", dropna=False):
            values = group[col].dropna().astype(float)
            if values.empty:
                continue
            rows.append(
                {
                    "event_label": str(label),
                    "horizon": horizon,
                    "p25": float(values.quantile(0.25)),
                    "mean": float(values.mean()),
                    "p75": float(values.quantile(0.75)),
                    "n": int(len(values)),
                }
            )
    plot_df = pd.DataFrame(rows)
    path = out_path / "mss_aligned_return_distribution.png"

    fig, ax = plt.subplots(figsize=(11, 7))
    for label, group in plot_df.groupby("event_label"):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], group["mean"], marker="o", label=f"{label} mean")
        ax.fill_between(group["horizon"], group["p25"], group["p75"], alpha=0.15)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Forward horizon (bars)")
    ax.set_ylabel("Aligned return")
    ax.set_title("MSS aligned-return distribution: P25 / mean / P75")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def create_cisd_distribution_plot(events: pd.DataFrame, out_dir: Path | str, horizons: list[int] | tuple[int, ...] | None = None) -> Path:
    """Plot CISD aligned-return P25/mean/P75 by open/extreme break-level type."""
    events = _as_pandas(events)
    required = {"event_type", "cisd_break_level_type"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    horizon_cols = [c for c in events.columns if c.startswith("aligned_return_")]
    if horizons is not None:
        wanted = {f"aligned_return_{int(h)}" for h in horizons}
        horizon_cols = [c for c in horizon_cols if c in wanted]
    if not horizon_cols:
        raise ValueError("events has no aligned_return_* columns")

    cisd = events[events["event_type"] == "cisd"].copy()
    if cisd.empty:
        raise ValueError("events has no CISD rows")

    rows = []
    for col in horizon_cols:
        horizon = int(col.rsplit("_", 1)[1])
        for label, group in cisd.groupby("cisd_break_level_type", dropna=False):
            values = group[col].dropna().astype(float)
            if values.empty:
                continue
            rows.append(
                {
                    "event_label": f"cisd/{label}",
                    "horizon": horizon,
                    "p25": float(values.quantile(0.25)),
                    "mean": float(values.mean()),
                    "p75": float(values.quantile(0.75)),
                    "n": int(len(values)),
                }
            )
    plot_df = pd.DataFrame(rows)
    path = out_path / "cisd_aligned_return_distribution.png"

    fig, ax = plt.subplots(figsize=(11, 7))
    for label, group in plot_df.groupby("event_label"):
        group = group.sort_values("horizon")
        ax.plot(group["horizon"], group["mean"], marker="o", label=f"{label} mean")
        ax.fill_between(group["horizon"], group["p25"], group["p75"], alpha=0.15)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Forward horizon (bars)")
    ax.set_ylabel("Aligned return")
    ax.set_title("CISD aligned-return distribution: P25 / mean / P75")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _session_volume_heatmap(data: pd.DataFrame, path: Path) -> Path:
    filtered = data[(data["event_type"] == "mss") if "event_type" in data.columns else True].dropna(subset=["event_session", "leg_volume_bucket"])
    rows = []
    for (session, volume_bucket), group in filtered.groupby(["event_session", "leg_volume_bucket"], dropna=False):
        weights = group["n"].astype(float)
        if weights.sum() <= 0:
            continue
        values = group["win_rate"].astype(float)
        rows.append({"event_session": str(session), "leg_volume_bucket": str(volume_bucket), "win_rate": float((values * weights).sum() / weights.sum())})
    heat = pd.DataFrame(rows)
    if heat.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No MSS session/volume data", ha="center", va="center")
    else:
        session_order = [s for s in ["asia", "london", "ny_am", "ny_pm", "other"] if s in set(heat["event_session"])]
        volume_order = [v for v in ["low", "normal", "high"] if v in set(heat["leg_volume_bucket"])]
        pivot = heat.pivot(index="event_session", columns="leg_volume_bucket", values="win_rate").reindex(index=session_order, columns=volume_order)
        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(pivot.to_numpy(dtype=float), vmin=0.4, vmax=0.6, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        ax.set_title("MSS win rate by session and leg-volume bucket")
        for y in range(len(pivot.index)):
            for x in range(len(pivot.columns)):
                val = pivot.iloc[y, x]
                if pd.notna(val):
                    ax.text(x, y, f"{val:.1%}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, label="Win rate")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def create_mss_distribution_plot_from_parquet(events_parquet: Path | str, out_dir: Path | str) -> Path:
    return create_mss_distribution_plot(pl.read_parquet(events_parquet), out_dir)


def create_cisd_distribution_plot_from_parquet(events_parquet: Path | str, out_dir: Path | str) -> Path:
    return create_cisd_distribution_plot(pl.read_parquet(events_parquet), out_dir)


def create_summary_plots_from_csv(summary_csv: Path | str, out_dir: Path | str) -> list[Path]:
    return create_summary_plots(pl.read_csv(summary_csv).to_pandas(), out_dir)
