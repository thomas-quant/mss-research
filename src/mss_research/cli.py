from __future__ import annotations

import argparse
from pathlib import Path

from .features import StudyConfig
from .pipeline import run_directory
from .plots import (
    create_cisd_distribution_plot,
    create_cisd_distribution_plot_from_parquet,
    create_mss_distribution_plot,
    create_mss_distribution_plot_from_parquet,
    create_summary_plots_from_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MSS/divergence event study.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run study over parquet files.")
    run.add_argument("--data", type=Path, default=Path("data"), help="Directory containing parquet OHLCV files.")
    run.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory.")
    run.add_argument("--timeframes", default="1min,5min,15min", help="Comma-separated pandas resample rules.")
    run.add_argument("--horizons", default="5,15,30,60", help="Comma-separated forward horizons in bars.")
    run.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for confidence intervals.")
    run.add_argument("--plots", action="store_true", help="Create matplotlib PNG charts after the study run.")
    plots = sub.add_parser("plots", help="Create PNG charts from a summary CSV.")
    plots.add_argument("--summary", type=Path, default=Path("outputs/summary.csv"), help="Summary CSV path.")
    plots.add_argument("--out", type=Path, default=Path("outputs/figures"), help="Figure output directory.")
    plots.add_argument("--events", type=Path, default=Path("outputs/all_events.parquet"), help="Events parquet path for distribution plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        config = StudyConfig(
            timeframes=tuple(x.strip() for x in args.timeframes.split(",") if x.strip()),
            horizons=tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip()),
            bootstrap_iterations=args.bootstrap,
        )
        events, summary = run_directory(args.data, args.out, config)
        if args.plots:
            figure_dir = args.out / "figures"
            paths = create_summary_plots_from_csv(args.out / "summary.csv", figure_dir)
            paths.append(create_mss_distribution_plot(events, figure_dir, horizons=config.horizons))
            paths.append(create_cisd_distribution_plot(events, figure_dir, horizons=config.horizons))
            print(f"plots={len(paths)} figures_dir={figure_dir}")
        print(f"events={len(events)} summary_rows={len(summary)} out={args.out}")
    elif args.command == "plots":
        paths = create_summary_plots_from_csv(args.summary, args.out)
        if args.events.exists():
            paths.append(create_mss_distribution_plot_from_parquet(args.events, args.out))
            paths.append(create_cisd_distribution_plot_from_parquet(args.events, args.out))
        print(f"plots={len(paths)} figures_dir={args.out}")
