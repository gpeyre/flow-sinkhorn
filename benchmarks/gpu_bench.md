# GPU Line-Benchmark Handoff (NeurIPS Supplementary)

This note is for an execution agent running the **line-segment benchmark on a GPU machine** to contrast with the current CPU benchmark.

The target is to reproduce the same benchmark logic and generate the corresponding figure, with one critical requirement:

> For the line graph benchmark, the unregularized reference \(W_1\) cost must use the **closed-form expression**, not a Floyd-Warshall-based computation.

---

## 1) Scope and expected output

- Benchmark: `line` only.
- Produce panel PDFs similar to the existing paper panels (graph + convergence).
- Use a distinct setup tag (recommended: `gpu`) so files do not overwrite existing `small` / `large` outputs.

Recommended expected figure names in `neurips/figures/`:

- `benchmark-gpu-line-graph.pdf`
- `benchmark-gpu-line-convergence.pdf`

---

## 2) Built-in options (already available)

The benchmark runner now exposes line-specific options:

- `--line-closed-form-w1`
  - Uses closed-form line \(W_1\) as the unregularized reference objective.
  - Automatically switches the ground cost geodesic matrix to analytic path distance \(D_{ij}=|i-j|\) (no Floyd-Warshall for line distance).
- `--line-analytic-distance`
  - Uses analytic path distance \(D_{ij}=|i-j|\) instead of Floyd-Warshall for line benchmarks.
- `--line-check-closed-form`
  - Runs strict sanity checks and prints diagnostics:
    - max distance-matrix error between Floyd-Warshall and analytic path distance
    - absolute error between closed-form \(W_1\) and LP \(W_1\)

Use these options for GPU line runs.

---

## 3) Closed-form reference for line graph

For a path graph with unit edge lengths and signed supply `z` (sum zero):

\[
W_1(z) = \sum_{i=1}^{n-1} \left| \sum_{j=1}^{i} z_j \right|.
\]

Implementation template:

```python
def line_w1_closed_form(z: np.ndarray) -> float:
    c = np.cumsum(np.asarray(z, dtype=np.float64))
    return float(np.sum(np.abs(c[:-1])))
```

For path metric distances, use analytic matrix:

```python
idx = np.arange(n, dtype=np.float64)
D = np.abs(idx[:, None] - idx[None, :])
```

No FW call is needed for line.

---

## 4) Execution checklist

1. Use line benchmark with:
   - `--line-closed-form-w1`
   - `--line-check-closed-form`
   - (`--line-analytic-distance` is optional in this mode, since `--line-closed-form-w1` already enforces analytic line distance)
2. Ensure GPU backend is active for the flow method (Torch/CUDA).
3. Keep benchmark comparability:
   - Same line graph generation logic (`n`, `m`, source/sink construction) as CPU benchmark.
   - Same gamma grids unless intentionally changed and documented.

---

## 5) Validation protocol (mandatory)

Before final figure generation:

1. Log and save:
   - `W1_closed_form`
   - `W1_lp` (optional cross-check only)
   - absolute/relative difference
2. Accept only if:
   - `abs(W1_closed_form - W1_lp) <= 1e-8` (or tight tolerance consistent with solver precision).
3. Confirm in logs:
   - `[check] line_distance_max_abs_err=...`
   - `[check] line_w1_abs_err=...`
   - GPU device used for flow benchmark (`cuda` visible in logs).

Recommended one-line assertion:

```python
assert abs(w1_ref_closed - lp_obj) <= 1e-8, "Closed-form and LP reference mismatch on line graph."
```

---

## 6) Suggested run command (after patch)

Example (adapt `--flow-max-iters` / `--vanilla-max-iters` to your budget):

```bash
MPLCONFIGDIR=/tmp/mpl \
python3 benchmarks/run_benchmark.py \
  --bench line \
  --setup-tag gpu \
  --n 80 \
  --m 6 \
  --flow-gammas 0.001 0.01 0.1 1.0 5.0 \
  --vanilla-gammas 0.005 0.02 0.08 0.3 \
  --line-closed-form-w1 \
  --line-check-closed-form \
  --flow-max-iters 12000 \
  --vanilla-max-iters 30000
```

---

## 7) Deliverables for merge

1. Generated figure files:
   - `neurips/figures/benchmark-gpu-line-graph.pdf`
   - `neurips/figures/benchmark-gpu-line-convergence.pdf`
2. A short run log (or markdown note) that includes:
   - GPU model + CUDA/PyTorch version
   - closed-form vs LP reference check numbers
   - final command used
3. If paper integration is requested:
   - add a dedicated caption sentence indicating line GPU benchmark and that line \(W_1\) reference uses closed form.
