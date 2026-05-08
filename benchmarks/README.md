# Benchmark Reproduction Guide

This directory contains the scripts used to generate the benchmark figures and tables for the supplementary material. The code is designed to be run from the repository root after installing the package and benchmark dependencies.

## What Is Reproduced

The unified runner `benchmarks/run_benchmark.py` supports three settings:

- `line`: path graph with analytic path distance and an optional closed-form unregularized W1 reference.
- `delaunay`: random planar Delaunay graph with source and target supports near opposite corners.
- `singlecell`: kNN graph built from the Waddington-OT tutorial dataset.

Each run writes:

- paper-panel files to `neurips/figures/benchmark-<setup-tag>-<bench>-graph.pdf` and `neurips/figures/benchmark-<setup-tag>-<bench>-convergence.pdf`;
- detailed diagnostic figures to `benchmarks/results/figures/`;
- CSV/LaTeX summary tables to `benchmarks/results/tables/`;
- an optional local PDF report to `benchmarks/results/report-<bench>.pdf` when `pdflatex` is available.

Generated outputs are not required to run the code and are not included in the source-only supplementary archive. The runner creates these directories locally when the benchmarks are rerun.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
pip install -r benchmarks/requirements.txt
```

For GPU runs, install a PyTorch build matching your CUDA driver before or instead of the generic `torch` wheel in `benchmarks/requirements.txt`.

## Data

### Synthetic Benchmarks

The `line` and `delaunay` benchmarks are self-contained and require no external data.

### Single-cell Benchmark

The single-cell benchmark uses the public Waddington-OT tutorial dataset. The dataset is not bundled in the archive. By default, files are searched recursively under `data/wot/`. The runner can download the tutorial archive using `gdown` if the files are missing.

Expected files after download/extraction:

- `ExprMatrix.var.genes.h5ad` or `ExprMatrix.h5ad`;
- `cell_days.txt`;
- optional `serum_cell_ids.txt`.

You can either let the code download the data on first use, or pre-populate `data/wot/` manually after extracting the archive. To use a different location, pass `--data-root /path/to/wot-data`.

## Smoke Tests

These commands are intentionally small and are suitable for checking that the environment works:

```bash
python benchmarks/run_benchmark.py \
  --bench line \
  --setup-tag smoke \
  --n 40 \
  --m 4 \
  --flow-gammas 0.01 0.1 \
  --vanilla-gammas 0.02 0.2 \
  --flow-max-iters 30 \
  --vanilla-max-iters 30 \
  --line-closed-form-w1 \
  --line-check-closed-form

python benchmarks/run_benchmark.py \
  --bench delaunay \
  --setup-tag smoke \
  --n 60 \
  --m 4 \
  --flow-gammas 0.002 0.006 \
  --vanilla-gammas 0.002 0.006 \
  --flow-max-iters 30 \
  --vanilla-max-iters 30
```

For a small single-cell smoke test after data are available:

```bash
python benchmarks/run_benchmark.py \
  --bench singlecell \
  --setup-tag smoke \
  --n 80 \
  --n0 20 \
  --k 4 \
  --pca 20 \
  --flow-gammas 0.3 1.0 \
  --vanilla-gammas 1.0 3.0 \
  --flow-max-iters 20 \
  --vanilla-max-iters 20
```

## Paper-style Runs

The parameters below match the documented small configurations used for paper-style figures. They are more expensive than the smoke tests.

```bash
python benchmarks/run_benchmark.py \
  --bench line \
  --setup-tag small \
  --n 80 \
  --m 6 \
  --flow-gammas 0.001 0.01 0.1 1.0 5.0 \
  --vanilla-gammas 0.005 0.02 0.08 0.3 \
  --flow-max-iters 1500 \
  --vanilla-max-iters 4000 \
  --line-closed-form-w1 \
  --line-check-closed-form

python benchmarks/run_benchmark.py \
  --bench delaunay \
  --setup-tag small \
  --n 140 \
  --m 7 \
  --flow-gammas 0.0009 0.002 0.0045 0.009 \
  --vanilla-gammas 0.0009 0.002 0.0045 0.009 \
  --flow-max-iters 2400 \
  --vanilla-max-iters 6000

python benchmarks/run_benchmark.py \
  --bench singlecell \
  --setup-tag small \
  --n 240 \
  --n0 60 \
  --k 4 \
  --pca 30 \
  --flow-gammas 0.3 1.0 3.0 9.0 \
  --vanilla-gammas 1.0 2.0 3.0 5.0 \
  --flow-max-iters 700 \
  --vanilla-max-iters 1400
```

Larger configurations are recorded in `benchmarks/benchmarks_parameters.md`.

## GPU Line Benchmark

Use GPU only when PyTorch sees a CUDA device. The runner can enforce this with `--require-gpu`:

```bash
python benchmarks/run_benchmark.py \
  --bench line \
  --setup-tag gpu \
  --n 80 \
  --m 6 \
  --flow-gammas 0.001 0.01 0.1 1.0 5.0 \
  --vanilla-gammas 0.005 0.02 0.08 0.3 \
  --line-closed-form-w1 \
  --line-check-closed-form \
  --use-torch \
  --device cuda \
  --require-gpu \
  --flow-max-iters 12000 \
  --vanilla-max-iters 30000
```

The line benchmark should always use `--line-closed-form-w1` for the unregularized reference. With `--line-check-closed-form`, the script prints and checks the analytic distance matrix, closed-form W1 objective, and closed-form dual potential against the LP reference on the same small instance.

## Reproducibility Notes

- `--seed` controls Delaunay sampling and single-cell subsampling.
- Synthetic graph generation is deterministic for a fixed seed and parameter set.
- Wall-clock timings depend on CPU/GPU hardware, BLAS, PyTorch, and solver versions; the main qualitative comparison should be reproduced from the curves and summaries, not from exact runtimes alone.
- The dense vanilla baseline uses all-pairs graph distances and can become memory-heavy for large graphs.
