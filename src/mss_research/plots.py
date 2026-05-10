from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl


def _as_pandas(df):
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


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


def create_summary_plots_from_csv(summary_csv: Path | str, out_dir: Path | str) -> list[Path]:
    return create_summary_plots(pl.read_csv(summary_csv).to_pandas(), out_dir)
