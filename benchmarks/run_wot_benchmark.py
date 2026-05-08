#!/usr/bin/env python3
"""CLI wrapper for single-cell WOT benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.wot_benchmark import (
    BenchmarkConfig,
    download_wot_tutorial_data,
    extract_zip,
    run_benchmark,
    summarize_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/wot"))
    p.add_argument("--download", action="store_true", help="Download WOT tutorial data if missing.")
    p.add_argument("--extract", action="store_true", help="Extract data.zip into data-root.")
    p.add_argument("--n0", type=int, default=200, help="Max sampled cells per timepoint.")
    p.add_argument("--pca", type=int, default=30, help="PCA dimensions.")
    p.add_argument("--k", type=int, default=4, help="k for k-NN graph.")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--niter", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-timepoints", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="Force CPU sparse solver.")
    p.add_argument("--device", type=str, default=None, help="Torch device (cuda/cpu/mps).")
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("benchmarks/results/summary.csv"),
        help="Where to save the summary table.",
    )
    p.add_argument(
        "--time-profile-csv",
        type=Path,
        default=Path("benchmarks/results/time_profile.csv"),
        help="Where to save empirical vs OT flux day profile.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.download:
        zip_path = download_wot_tutorial_data(args.data_root)
        print(f"Downloaded: {zip_path}")

    if args.extract:
        zip_path = args.data_root / "data.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing {zip_path}. Use --download first.")
        out_dir = extract_zip(zip_path, args.data_root)
        print(f"Extracted into: {out_dir}")

    cfg = BenchmarkConfig(
        data_root=args.data_root,
        random_state=args.seed,
        n_cells_per_time=args.n0,
        pca_components=args.pca,
        knn_k=args.k,
        epsilon=args.epsilon,
        niter=args.niter,
        use_torch=not args.cpu,
        device=args.device,
        max_timepoints=args.max_timepoints,
    )

    result = run_benchmark(cfg)
    summary = summarize_benchmark(result)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.time_profile_csv.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(args.summary_csv, index=False)
    result["time_profile"].to_csv(args.time_profile_csv)

    print("\nBenchmark summary")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {args.summary_csv}")
    print(f"Saved time profile to: {args.time_profile_csv}")


if __name__ == "__main__":
    main()
