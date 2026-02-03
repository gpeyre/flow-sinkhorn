# Flow Sinkhorn

<p align="center">
  <img src="logo/flow-sinkhorn.jpg" alt="Flow Sinkhorn logo" width="60%">
</p>

This repository contains the reference implementation of the **Flow Sinkhorn** algorithm for computing approximate Wasserstein-1 distances on graphs.

The algorithm is introduced and analyzed in the paper:

> **Robust Sublinear Convergence Rates for Iterative Bregman Projections on Affine Spaces**
> Gabriel Peyré
> *arXiv preprint, 2026*
> https://arxiv.org/abs/2602.01372

Flow Sinkhorn can be seen as a flow-based interpretation and implementation of Sinkhorn-type iterations, with strong robustness and convergence guarantees derived from the theory of iterative Bregman projections.

---

## Installation

### Basic installation

```bash
pip install -e .
```

### With optional dependencies

For sparse graph support:
```bash
pip install -e ".[sparse]"
```

For exact solver (linear programming):
```bash
pip install -e ".[exact]"
```

For **GPU acceleration** with PyTorch:
```bash
pip install -e ".[gpu]"
```

For running examples and notebooks:
```bash
pip install -e ".[examples]"
```

Install everything (including GPU support):
```bash
pip install -e ".[all]"
```

---

## Quick Start

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

---

## Package Structure

```
flow-sinkhorn/
├── flowsinkhorn/           # Main Python package
│   ├── __init__.py        # Package initialization with main exports
│   ├── sinkhorn.py        # Sinkhorn algorithms (NumPy/SciPy, CPU)
│   ├── sinkhorn_torch.py  # GPU-accelerated Sinkhorn (PyTorch)
│   └── exact.py           # Exact LP solver (CVXPY)
├── examples/              # Example notebooks
│   ├── planar-graph.ipynb   # K-NN graph example
│   ├── mesh-transport.ipynb # 3D mesh example
│   ├── grid-transport.ipynb # 2D grid with obstacles
│   └── gpu-benchmark.ipynb  # GPU acceleration benchmark
├── paper/                 # LaTeX paper
│   └── flow-sinkhorn.tex
├── setup.py              # Installation script
└── README.md             # This file
```

---

## Documentation

### Main Functions

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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpeyre/flow-sinkhorn/blob/main/examples/planar-graph.ipynb)

#### 2. **3D Mesh Example** (`mesh-transport.ipynb`)
- Loading and visualizing 3D meshes (OFF format)
- Building graph from mesh edges
- Selecting sources (top vertices) and sinks (bottom vertices)
- Computing exact and approximate optimal transport
- Comparing different regularization levels
- 3D visualization of transport flows on mesh

This example showcases optimal transport on real 3D geometry using the `data/moomoo.off` mesh.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpeyre/flow-sinkhorn/blob/main/examples/mesh-transport.ipynb)

#### 3. **2D Grid with Obstacles** (`grid-transport.ipynb`)
- Creating regular 30×30 square grid graphs
- Modulating edge costs with Gaussian bumps (obstacles)
- Corner-to-corner transport (top-left to bottom-right)
- Visualizing obstacle avoidance behavior
- Comparing exact vs Sinkhorn with different regularizations
- 2D heatmap visualization with flow overlays

This example illustrates **path planning** and **obstacle avoidance** in optimal transport.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpeyre/flow-sinkhorn/blob/main/examples/grid-transport.ipynb)

#### 4. **GPU Benchmark** (`gpu-benchmark.ipynb`)
- GPU/CPU device detection (CUDA, MPS, CPU)
- PyTorch GPU-accelerated implementation
- **Numerical equivalence verification** (NumPy vs PyTorch, machine precision)
- **Wall-clock time benchmarking** for fixed iterations
- Performance scaling vs graph size
- Speedup measurements (10-100x typical on GPU)

This example validates the PyTorch implementation and demonstrates GPU acceleration benefits.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpeyre/flow-sinkhorn/blob/main/examples/gpu-benchmark.ipynb)

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
\min_{f \geq 0} \langle f, W \rangle + \varepsilon \cdot \text{KL}(f | 1) \quad \text{s.t.} \quad f^\top \mathbf{1} - f \mathbf{1} = z
$$

where:
- $f_{i,j}$ is the flow from node $j$ to node $i$
- $W_{i,j}$ is the cost of transporting mass along edge $(j,i)$
- $z_i$ is the source/sink at node $i$ (sum to 0)
- $\varepsilon > 0$ is the entropic regularization parameter

### Iterative Updates

The algorithm iteratively computes the flow from dual potentials $h$:

$$
f_{i,j} = \exp\left(\frac{-W_{i,j} + h_i - h_j}{\varepsilon}\right)
$$

The potentials are updated using:

$$
h \leftarrow \frac{h}{2} - \frac{\varepsilon}{2} m
$$

where the update vector $m$ is computed from auxiliary variables:

$$
a_i = \sum_j \exp\left(\frac{-W_{i,j} - h_i}{\varepsilon}\right), \quad
b_i = \sum_j \exp\left(\frac{-W_{i,j} + h_i}{\varepsilon}\right)
$$

For numerical stability, $m$ is computed differently depending on the node type (with $r = z/2$):

- **Neutral nodes** ($z_i = 0$): $m_i = \frac{1}{2}\left(\log a_i - \log b_i\right)$

- **Source nodes** ($z_i > 0$): $m_i = \log\left(\sqrt{r_i^2 + a_i b_i} + r_i\right) - \log b_i$

- **Sink nodes** ($z_i < 0$): $m_i = -\log\left(\sqrt{r_i^2 + a_i b_i} - r_i\right) + \log a_i$

These formulas ensure numerical stability and enforce the flow conservation constraint $f^\top \mathbf{1} - f \mathbf{1} = z$.

See the paper for complete derivation, convergence analysis, and theoretical guarantees.

---

## Performance

The sparse implementation is particularly efficient for graphs with $O(n)$ edges (e.g., K-NN graphs, grid graphs) rather than $O(n^2)$ edges:

- **Dense version**: Suitable for small graphs or dense cost matrices
- **Sparse version**: Recommended for large graphs with sparse structure (can be 10-100x faster)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{peyre2026flowsinkhorn,
  title={Robust Sublinear Convergence Rates for Iterative Bregman Projections on Affine Spaces},
  author={Peyr{\'e}, Gabriel},
  journal={arXiv preprint arXiv:2602.01372},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

Gabriel Peyré
CNRS and ENS, Université PSL
gabriel.peyre@ens.fr
