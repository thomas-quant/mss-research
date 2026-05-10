from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _clean_summary(summary: pd.DataFrame) -> pd.DataFrame:
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


def create_mss_distribution_plot(events: pd.DataFrame, out_dir: Path | str, horizons: list[int] | tuple[int, ...] | None = None) -> Path:
    """Plot MSS aligned-return P25/mean/P75 by horizon and structure subtype."""
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


def create_mss_distribution_plot_from_parquet(events_parquet: Path | str, out_dir: Path | str) -> Path:
    return create_mss_distribution_plot(pd.read_parquet(events_parquet), out_dir)


def create_summary_plots_from_csv(summary_csv: Path | str, out_dir: Path | str) -> list[Path]:
    return create_summary_plots(pd.read_csv(summary_csv), out_dir)
