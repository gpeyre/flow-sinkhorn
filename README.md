# Flow Sinkhorn

<p align="center">
  <img src="logo/flow-sinkhorn.jpg" alt="Flow Sinkhorn logo" width="60%">
</p>

This repository contains:
- the reference implementation of **Flow Sinkhorn** for approximate Wasserstein-1 on graphs,
- benchmark and example code for reproducing the numerical experiments,
- and a Lean formalization of the main convergence/complexity blueprint.

The algorithm is introduced and analyzed in the paper:

> **Robust Sublinear Convergence Rates for Iterative Bregman Projections**
> Gabriel Peyré
> *Preprint, 2026*

Flow Sinkhorn can be seen as a flow-based interpretation and implementation of Sinkhorn-type iterations, with strong robustness and convergence guarantees derived from the theory of iterative Bregman projections.

---

## Installation

For a local checkout, install the package in editable mode from the repository root:

```bash
pip install -e .
```

Optional dependency groups can be installed depending on what you want to run:

```bash
pip install -e ".[sparse]"      # sparse graph support via sparse.COO
pip install -e ".[exact]"       # exact LP references via CVXPY
pip install -e ".[gpu]"         # PyTorch implementation
pip install -e ".[examples]"    # notebooks and plotting dependencies
pip install -e ".[benchmarks]"  # benchmark runner dependencies
pip install -e ".[all]"         # everything above
```

For benchmark reproduction, the equivalent explicit install is:

```bash
pip install -r requirements.txt
pip install -r benchmarks/requirements.txt
```

For CUDA runs, install the PyTorch wheel matching the target machine before running GPU benchmarks. On macOS/Apple Silicon, a conda environment is often the most robust option:

```bash
conda create -n flowsinkhorn python=3.10
conda activate flowsinkhorn
pip install -e ".[all]"
```

Quick installation check:

```bash
python - <<'PY'
import numpy as np
from flowsinkhorn import sinkhorn_w1
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
W = 1 / (A + 1e-9)
z = np.array([1.0, 0.0, -1.0])
f, err, h = sinkhorn_w1(W, z, epsilon=0.1, niter=100)
print(f"Final error: {err[-1]:.2e}")
PY
```

## Quick Start

A minimal dense example on a three-node chain:

```python
import numpy as np
from flowsinkhorn import sinkhorn_w1, solve_w1_exact

# Create a simple graph
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
W = 1 / (A + 1e-9)  # Cost matrix (large for non-edges)
z = np.array([1.0, 0.0, -1.0])  # Source/sink vector

# Approximate solution (fast, entropic regularization)
f, err, h = sinkhorn_w1(W, z, epsilon=0.1, niter=100)

# Exact solution (slower, linear programming)
F, obj_val, status = solve_w1_exact(W, z)
```

Use the dense solver for small dense cost matrices, `sinkhorn_w1_sparse` for large sparse graphs such as K-NN graphs or meshes, and `solve_w1_exact` only for small ground-truth references because it solves a linear program.

---

## Package Structure

```
flow-sinkhorn/
├── benchmarks/            # Benchmark scripts
├── data/                  # Local datasets (ignored by git)
├── examples/              # Example notebooks
├── flowsinkhorn/          # Main Python package
├── lean/                  # Lean formalization
├── setup.py               # Installation script
└── README.md              # This file
```

---

## Formal Verification (Lean)

The machine-checked formalization is in `lean/`.

- Canonical umbrella import: `FlowSinkhorn.KLProjection`
- Full certification-chain import: `FlowSinkhorn.KLProjection.Certification`
- Status and audit map: [`lean/README.md`](lean/README.md)

Quick verification:

```bash
cd lean
lake build
rg '^\s*theorem\b' FlowSinkhorn/KLProjection | wc -l
rg '^\s*(def|structure)\b' FlowSinkhorn/KLProjection | wc -l
rg '^\s*(sorry|admit|axiom)\b' FlowSinkhorn/KLProjection
```

The paper appendix explains the paper-label to Lean-constant map and the
certification workflow. The paper source itself is not required to run the code
or verify the Lean project.

---

## Documentation

### Main Functions

The Python API keeps the historical keyword `epsilon`; it is the regularization parameter denoted $\gamma$ in the paper.

#### `sinkhorn_w1(W, z, epsilon, niter)`
Entropic regularized W1 optimal transport solver (dense matrices).

**Parameters:**
- `W`: Cost/distance matrix (n×n)
- `z`: Source/sink vector (n,) with sum(z) = 0
- `epsilon`: Regularization parameter (smaller = more accurate but slower)
- `niter`: Number of iterations

**Returns:**
- `f`: Flow matrix (n×n)
- `err`: List of constraint errors at each iteration
- `h`: Dual variable (potentials)

#### `sinkhorn_w1_sparse(W, z, epsilon, niter)`
Sparse version of `sinkhorn_w1` for graphs with few edges.

**Parameters:** Same as `sinkhorn_w1`, but `W` should be a `sparse.COO` matrix.

**Returns:** Same as `sinkhorn_w1`, but `f` is a sparse matrix.

#### `sinkhorn_w1_torch(W, z, epsilon, niter, device=None)`
**GPU-accelerated** W1 Sinkhorn using PyTorch (requires PyTorch installation).

**Parameters:** Same as `sinkhorn_w1`, plus:
- `device`: 'cpu', 'cuda', 'mps', or None (auto-select)

**Returns:** Same as `sinkhorn_w1`

**Performance:** 10-100x speedup on GPU for large graphs (n > 1000).

#### `solve_w1_exact(W, z, solver=None, verbose=False)`
Exact W1 optimal transport solver using linear programming.

**Parameters:**
- `W`: Cost/distance matrix (n×n)
- `z`: Source/sink vector (n,)
- `solver`: CVXPY solver name (optional)
- `verbose`: Print solver output

**Returns:**
- `F`: Optimal flow matrix (n×n)
- `objective_value`: Optimal transport cost
- `status`: Solver status

---

## Examples

### Jupyter Notebooks

Four complete example notebooks are provided:

#### 1. **Planar Graph Example** (`planar-graph.ipynb`)
- Graph generation (K-NN graphs)
- Exact solver using linear programming
- Dense and sparse Sinkhorn implementations
- Convergence analysis
- Effect of regularization parameter
- Visualization of flows

Run locally from `examples/planar-graph.ipynb` or upload the supplementary archive to Colab.

#### 2. **3D Mesh Example** (`mesh.ipynb`)
- Loading and visualizing 3D meshes (OFF format)
- Building graph from mesh edges
- Selecting sources (top vertices) and sinks (bottom vertices)
- Computing exact and approximate optimal transport
- Comparing different regularization levels
- 3D visualization of transport flows on mesh

This example showcases optimal transport on real 3D geometry using the `data/moomoo.off` mesh.

Run locally from `examples/mesh.ipynb` or upload the supplementary archive to Colab.

#### 3. **2D Grid with Obstacles** (`grid.ipynb`)
- Creating regular 30×30 square grid graphs
- Modulating edge costs with Gaussian bumps (obstacles)
- Corner-to-corner transport (top-left to bottom-right)
- Visualizing obstacle avoidance behavior
- Comparing exact vs Sinkhorn with different regularizations
- 2D heatmap visualization with flow overlays

This example illustrates **path planning** and **obstacle avoidance** in optimal transport.

Run locally from `examples/grid.ipynb` or upload the supplementary archive to Colab.

#### 4. **GPU Benchmark** (`gpu-benchmark.ipynb`)
- GPU/CPU device detection (CUDA, MPS, CPU)
- PyTorch GPU-accelerated implementation
- **Numerical equivalence verification** (NumPy vs PyTorch, machine precision)
- **Wall-clock time benchmarking** for fixed iterations
- Performance scaling vs graph size
- Speedup measurements (10-100x typical on GPU)

This example validates the PyTorch implementation and demonstrates GPU acceleration benefits.

Run locally from `examples/gpu-benchmark.ipynb` or upload the supplementary archive to Colab.

**Run the notebooks:**

```bash
cd examples
jupyter notebook  # Opens Jupyter browser
```

For GPU acceleration:
```bash
pip install -e ".[gpu]"  # Install PyTorch
jupyter notebook gpu-benchmark.ipynb
```

---

## Algorithm Overview

The Flow Sinkhorn algorithm solves the entropic regularized W1 optimal transport problem:

$$
\min_{f \geq 0} \langle f, W \rangle + \gamma \cdot \text{KL}(f | 1) \quad \text{s.t.} \quad f^\top \mathbf{1} - f \mathbf{1} = z
$$

where:
- $f_{i,j}$ is the flow from node $j$ to node $i$
- $W_{i,j}$ is the cost of transporting mass along edge $(j,i)$
- $z_i$ is the source/sink at node $i$ (sum to 0)
- $\gamma > 0$ is the entropic regularization parameter

### Iterative Updates

The algorithm iteratively computes the flow from dual potentials $h$:

$$
f_{i,j} = \exp\left(\frac{-W_{i,j} + h_i - h_j}{\gamma}\right)
$$

The potentials are updated using:

$$
h \leftarrow \frac{h}{2} - \frac{\gamma}{2} m
$$

where the update vector $m$ is computed from auxiliary variables:

$$
a_i = \sum_j \exp\left(\frac{-W_{i,j} - h_i}{\gamma}\right), \quad
b_i = \sum_j \exp\left(\frac{-W_{i,j} + h_i}{\gamma}\right)
$$

For numerical stability, $m$ is computed differently depending on the node type (with $r = z/2$):

- **Neutral nodes** ($z_i = 0$): $m_i = \frac{1}{2}\left(\log a_i - \log b_i\right)$

- **Source nodes** ($z_i > 0$): $m_i = \log\left(\sqrt{r_i^2 + a_i b_i} + r_i\right) - \log b_i$

- **Sink nodes** ($z_i < 0$): $m_i = -\log\left(\sqrt{r_i^2 + a_i b_i} - r_i\right) + \log a_i$

These formulas ensure numerical stability and enforce the flow conservation constraint $f^\top \mathbf{1} - f \mathbf{1} = z$.

See the submitted manuscript for the complete derivation, convergence analysis,
and theoretical guarantees.

---

## Performance

The sparse implementation is particularly efficient for graphs with $O(n)$ edges (e.g., K-NN graphs, grid graphs) rather than $O(n^2)$ edges:

- **Dense version**: Suitable for small graphs or dense cost matrices
- **Sparse version**: Recommended for large graphs with sparse structure (can be 10-100x faster)

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'sparse'`: install sparse support with `pip install -e ".[sparse]"` or `pip install sparse`.
- `ModuleNotFoundError: No module named 'cvxpy'`: install exact-solver support with `pip install -e ".[exact]"` or `pip install cvxpy`.
- Slow convergence: increase the regularization parameter (`epsilon` in the Python API, $\gamma$ in the paper), increase `niter`, or switch to the sparse solver when the graph is sparse.
- Numerical issues: check that the cost matrix has finite, well-scaled values on valid edges; the implementation uses stabilized log-sum-exp updates but extremely small regularization can still be ill-conditioned.
- Uninstall with `pip uninstall flowsinkhorn`.

---

## Development and Contributing

Contributions are welcome. A typical development setup is:

```bash
git clone https://github.com/gpeyre/flow-sinkhorn.git
cd flow-sinkhorn
pip install -e ".[all]"
git checkout -b feature/your-feature-name
```

Guidelines:

- Keep functions focused and documented with NumPy-style docstrings.
- Prefer descriptive variable names and type hints where they clarify intent.
- Preserve backward compatibility of the public Python API when possible.
- Before opening a pull request, run a small Python smoke test, check relevant notebooks or benchmarks, and verify Lean changes with `lake build` from `lean/` when touching formalization files.
- Pull requests should explain the motivation, summarize the change, and mention any relevant numerical or formal verification checks.

Areas especially useful for contribution include performance improvements, additional graph/OT examples, stronger tests, documentation, benchmark extensions, and Lean formalization cleanup.

---

## Citation

Please cite the paper once the public bibliographic entry is available. For now,
refer to:

```bibtex
@misc{peyre2026flowsinkhorn,
  title={Robust Sublinear Convergence Rates for Iterative Bregman Projections},
  author={Peyr{\'e}, Gabriel},
  year={2026},
  note={Preprint}
}
```

---

## License

See `LICENSE`.

---

## Contact

Gabriel Peyré, `gabriel.peyre@ens.fr`.
