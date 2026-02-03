"""
Flow Sinkhorn: Flow-based Sinkhorn algorithm for W1 optimal transport.

This package provides efficient implementations of the Sinkhorn-flow algorithm
for computing approximate Wasserstein-1 distances on graphs, along with exact
solvers for reference.

Main components
---------------
- sinkhorn: Entropic regularized solvers (NumPy/SciPy, CPU)
- sinkhorn_torch: GPU-accelerated solvers (PyTorch, optional)
- exact: Exact linear programming solvers (CVXPY, slower, exact)
- utils: Utility functions for graph manipulation and visualization

Quick Start
-----------
>>> import numpy as np
>>> from flowsinkhorn import sinkhorn_w1, solve_w1_exact
>>>
>>> # Create a simple graph
>>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
>>> W = 1 / (A + 1e-9)  # Cost matrix
>>> z = np.array([1.0, 0.0, -1.0])  # Source/sink
>>>
>>> # Approximate solution (fast)
>>> f, err, h = sinkhorn_w1(W, z, epsilon=0.1, niter=100)
>>>
>>> # Exact solution (slower)
>>> F, obj_val, status = solve_w1_exact(W, z)

References
----------
Gabriel Peyré, "Robust Sublinear Convergence Rates for Iterative Bregman
Projections on Affine Spaces", arXiv preprint, 2026.
https://arxiv.org/abs/2602.01372
"""

__version__ = "0.1.0"
__author__ = "Gabriel Peyré"

# Import main functions for convenient access
from .sinkhorn import sinkhorn_w1, sinkhorn_w1_sparse
from .exact import solve_w1_exact, solve_w1_exact_sparse

# Utility functions (always available)
from .utils import (
    load_off_file,
    build_mesh_graph,
    select_sources_sinks,
    plot_mesh,
    plot_mesh_with_flow
)

# PyTorch implementation (optional, only if torch is available)
try:
    from .sinkhorn_torch import (
        sinkhorn_w1_torch,
        sinkhorn_w1_torch_sparse,
        check_gpu_availability,
        get_device
    )
    _torch_available = True
except ImportError:
    _torch_available = False

__all__ = [
    'sinkhorn_w1',
    'sinkhorn_w1_sparse',
    'solve_w1_exact',
    'solve_w1_exact_sparse',
    'load_off_file',
    'build_mesh_graph',
    'select_sources_sinks',
    'plot_mesh',
    'plot_mesh_with_flow',
]

# Add PyTorch functions if available
if _torch_available:
    __all__.extend([
        'sinkhorn_w1_torch',
        'sinkhorn_w1_torch_sparse',
        'check_gpu_availability',
        'get_device',
    ])
