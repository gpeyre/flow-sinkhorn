# Installation Guide

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/gpeyre/flow-sinkhorn.git
cd flow-sinkhorn
```

### 2. Install the package

#### Option A: Basic installation (minimal dependencies)
```bash
pip install -e .
```

This installs only `numpy` and `scipy`, which are sufficient for the basic Sinkhorn algorithm.

#### Option B: Install with sparse support
```bash
pip install -e ".[sparse]"
```

Adds the `sparse` package for efficient sparse matrix operations.

#### Option C: Install with exact solver
```bash
pip install -e ".[exact]"
```

Adds `cvxpy` for the exact linear programming solver.

#### Option D: Install everything (recommended for development)
```bash
pip install -e ".[all]"
```

Installs all dependencies including:
- Core: `numpy`, `scipy`
- Sparse: `sparse`
- Exact: `cvxpy`
- Examples: `matplotlib`, `networkx`, `scikit-learn`, `jupyter`

### 3. Verify installation

```python
import numpy as np
from flowsinkhorn import sinkhorn_w1, solve_w1_exact

# Create a simple test
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
W = 1 / (A + 1e-9)
z = np.array([1.0, 0.0, -1.0])

# Run Sinkhorn
f, err, h = sinkhorn_w1(W, z, epsilon=0.1, niter=100)
print(f"Final error: {err[-1]:.2e}")
```

## Running Examples

After installation with `[examples]` or `[all]`:

```bash
cd examples
jupyter notebook planar-graph.ipynb
```

## Alternative: Using requirements.txt

If you prefer using `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install all dependencies at once.

## Troubleshooting

### ImportError: No module named 'sparse'

Install the sparse package:
```bash
pip install sparse
```

### ImportError: No module named 'cvxpy'

Install CVXPY:
```bash
pip install cvxpy
```

For better performance with CVXPY, you may also want to install additional solvers like MOSEK or Gurobi (requires licenses).

### On macOS with M1/M2 chip

Some packages may require special handling. Try using conda:
```bash
conda create -n flowsinkhorn python=3.10
conda activate flowsinkhorn
pip install -e ".[all]"
```

## Development Setup

For development, install in editable mode with all dependencies:

```bash
pip install -e ".[all]"
```

This allows you to modify the source code and see changes immediately without reinstalling.

## Uninstallation

```bash
pip uninstall flowsinkhorn
```
