# Flow Sinkhorn Package

This is the main Python package for Flow Sinkhorn algorithm.

## Modules

### `sinkhorn.py`
Implementation of the entropic regularized W1 optimal transport solver:
- `sinkhorn_w1()` - Dense version for general cost matrices
- `sinkhorn_w1_sparse()` - Sparse version for graphs with few edges

Both functions use stabilized log-sum-exp operations to avoid numerical overflow/underflow.

### `exact.py`
Exact solver using linear programming (requires CVXPY):
- `solve_w1_exact()` - Solves the Beckmann transshipment problem
- `solve_w1_exact_sparse()` - Edge-based formulation for sparse graphs

### `sinkhorn_torch.py`
GPU-accelerated solver using PyTorch (requires PyTorch):
- `sinkhorn_w1_torch()` - GPU-accelerated version with auto device selection
- `sinkhorn_w1_torch_sparse()` - Sparse GPU version
- `check_gpu_availability()` - Check available devices
- `get_device()` - Get best available device

### `utils.py`
Compatibility layer re-exporting mesh helpers from `toolbox.mesh`.
Prefer importing from `flowsinkhorn.toolbox`.

### `toolbox/`
Notebook-ready helper functions organized by theme:
- `toolbox.grid` - Grid graph creation, Gaussian obstacle costs, corner source/sink builders, and 2D plotting helpers
- `toolbox.planar` - Planar K-NN graph creation and normalized source/sink generation
- `toolbox.mesh` - OFF loading, mesh graph creation, source/sink selection, and 3D plotting helpers

Example:
```python
from flowsinkhorn.toolbox import create_grid_graph, compute_modulated_costs
```

### `__init__.py`
Package initialization that exports the main functions for convenient access:
```python
from flowsinkhorn import sinkhorn_w1, solve_w1_exact, plot_mesh
```

## Implementation Details

### Numerical Stability

The Sinkhorn algorithm uses several techniques for numerical stability:

1. **Log-sum-exp**: All exponentials are computed in log-domain using scipy's `logsumexp`
2. **Case distinction**: Three separate formulas for neutral nodes (z=0), sources (z>0), and sinks (z<0)
3. **Sparse operations**: For sparse graphs, only edge variables are stored and computed

### Sparse vs Dense

Use the sparse version when:
- Your graph has O(n) edges rather than O(n²)
- Memory is a constraint for large n
- You want faster iterations (10-100x speedup possible)

Use the dense version when:
- The graph is small (n < 1000)
- The cost matrix is dense
- You don't have the `sparse` package installed

## Dependencies

**Required:**
- numpy >= 1.18.0
- scipy >= 1.4.0

**Optional:**
- sparse >= 0.13.0 (for `sinkhorn_w1_sparse`)
- cvxpy >= 1.1.0 (for exact solvers)
- torch >= 1.9.0 (for GPU acceleration)
- matplotlib >= 3.1.0 (for visualization utilities)

## Algorithm Reference

The Flow Sinkhorn algorithm is described in:

> Gabriel Peyré, "Robust Sublinear Convergence Rates for Iterative Bregman
> Projections on Affine Spaces", arXiv:2602.01372, 2026.
