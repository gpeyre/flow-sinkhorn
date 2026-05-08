# Benchmark setups

This file records the benchmark parameters currently used for the paper-style figure generation.

## Medium benchmark

### Line graph
- Graph size: `n=80` nodes (path graph), support size `m=6`.
- Flow-Sinkhorn gammas: `[0.001, 0.01, 0.1, 1.0, 5.0]`.
- Vanilla Sinkhorn gammas: `[0.005, 0.02, 0.08, 0.3]`.
- Iteration budgets: `flow-max-iters=1500`, `vanilla-max-iters=4000`.

### Delaunay graph
- Graph size: `n=140` points, support size `m=7`.
- Flow-Sinkhorn gammas: `[0.0009, 0.002, 0.0045, 0.009]`.
- Vanilla Sinkhorn gammas: `[0.0009, 0.002, 0.0045, 0.009]`.
- Iteration budgets: `flow-max-iters=2400`, `vanilla-max-iters=6000`.

### Single-cell graph
- Graph size: target `n=240` cells.
- Construction parameters: `n0=60` cells per snapshot, `k=4`, `pca=30`.
- Flow-Sinkhorn gammas: `[0.3, 1.0, 3.0, 9.0]`.
- Vanilla Sinkhorn gammas: `[1.0, 2.0, 3.0, 5.0]`.
- Iteration budgets: `flow-max-iters=700`, `vanilla-max-iters=1400`.

## Large benchmark (2x graph size)

### Line graph
- Graph size: `n=160` nodes (path graph), support size `m=12`.
- Flow-Sinkhorn gammas: `[0.001, 0.01, 0.1, 1.0, 5.0]`.
- Vanilla Sinkhorn gammas: `[0.005, 0.02, 0.08, 0.3]`.
- Iteration budgets used for completion: `flow-max-iters=1500`, `vanilla-max-iters=4000`.

### Delaunay graph
- Graph size: `n=280` points, support size `m=14`.
- Flow-Sinkhorn gammas: `[0.0009, 0.002, 0.0045, 0.009]`.
- Vanilla Sinkhorn gammas: `[0.0009, 0.002, 0.0045, 0.009]`.
- Iteration budgets: `flow-max-iters=2400`, `vanilla-max-iters=6000`.

### Single-cell graph
- Graph size: target `n=480` cells.
- Construction parameters: `n0=120` cells per snapshot, `k=4`, `pca=30`.
- Flow-Sinkhorn gammas: `[0.3, 1.0, 3.0, 9.0]`.
- Vanilla Sinkhorn gammas: `[1.0, 2.0, 3.0, 5.0]`.
- Iteration budgets: `flow-max-iters=700`, `vanilla-max-iters=1400`.

## Figure files

The paper panel PDFs are available in `neurips/figures/` with a consistent naming:

- `benchmark-small-line-graph.pdf`
- `benchmark-small-line-convergence.pdf`
- `benchmark-small-delaunay-graph.pdf`
- `benchmark-small-delaunay-convergence.pdf`
- `benchmark-small-singlecell-graph.pdf`
- `benchmark-small-singlecell-convergence.pdf`
- `benchmark-large-line-graph.pdf`
- `benchmark-large-line-convergence.pdf`
- `benchmark-large-delaunay-graph.pdf`
- `benchmark-large-delaunay-convergence.pdf`
- `benchmark-large-singlecell-graph.pdf`
- `benchmark-large-singlecell-convergence.pdf`
