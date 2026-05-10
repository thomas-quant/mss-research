from __future__ import annotations

import argparse
from pathlib import Path

from .features import StudyConfig
from .pipeline import run_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MSS/divergence event study.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run study over parquet files.")
    run.add_argument("--data", type=Path, default=Path("data"), help="Directory containing parquet OHLCV files.")
    run.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory.")
    run.add_argument("--timeframes", default="1min,5min,15min", help="Comma-separated pandas resample rules.")
    run.add_argument("--horizons", default="5,15,30,60", help="Comma-separated forward horizons in bars.")
    run.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for confidence intervals.")
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
        print(f"events={len(events)} summary_rows={len(summary)} out={args.out}")
