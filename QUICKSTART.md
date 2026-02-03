# Quick Start Guide

Get started with Flow Sinkhorn in 5 minutes!

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/gpeyre/flow-sinkhorn.git
cd flow-sinkhorn

# Install with all dependencies
pip install -e ".[all]"
```

## 2. Basic Usage

### Python API

```python
import numpy as np
from flowsinkhorn import sinkhorn_w1, solve_w1_exact

# Create a simple 3-node chain graph
A = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]])

# Cost matrix (inverse of adjacency)
W = 1 / (A + 1e-9)

# Source at node 0, sink at node 2
z = np.array([1.0, 0.0, -1.0])

# Approximate solution (fast)
f, err, h = sinkhorn_w1(W, z, epsilon=0.1, niter=100)
print(f"Final error: {err[-1]:.2e}")

# Exact solution (slower)
F, cost, status = solve_w1_exact(W, z)
print(f"Optimal cost: {cost:.6f}")
```

## 3. Run Examples

### K-NN Graph Example

```bash
cd examples
jupyter notebook flow-sinkhorn.ipynb
```

**What it shows:**
- Random planar graphs
- Dense vs sparse implementations
- Convergence analysis
- Effect of regularization parameter

### 3D Mesh Example

```bash
cd examples
jupyter notebook mesh-transport.ipynb
```

**What it shows:**
- Loading 3D meshes (OFF format)
- Optimal transport on mesh graphs
- 3D flow visualization
- Comparison exact vs Sinkhorn

## 4. Key Functions

### `sinkhorn_w1(W, z, epsilon, niter)`
Entropic regularized solver (dense).
- **W**: Cost matrix (n×n)
- **z**: Source/sink vector (sum to 0)
- **epsilon**: Regularization (smaller = more accurate)
- **niter**: Number of iterations

**Returns:** flow matrix, errors, potentials

### `sinkhorn_w1_sparse(W, z, epsilon, niter)`
Same as above but for sparse graphs (faster, less memory).
- **W**: sparse.COO matrix

### `solve_w1_exact(W, z)`
Exact solution using linear programming.
- Slower but exact
- Requires CVXPY

## 5. When to Use What

**Use Dense Sinkhorn when:**
- Small graphs (n < 1000)
- Dense cost matrices
- Quick prototyping

**Use Sparse Sinkhorn when:**
- Large sparse graphs (e.g., K-NN, meshes)
- Memory is limited
- Need speed (10-100x faster)

**Use Exact Solver when:**
- Need ground truth reference
- Small problems (n < 500)
- Accuracy is critical

## 6. Common Patterns

### Sparse Graph (K-NN)

```python
import sparse
from sklearn.neighbors import NearestNeighbors

# Build K-NN graph
nbrs = NearestNeighbors(n_neighbors=5).fit(X.T)
distances, indices = nbrs.kneighbors(X.T)

# Create sparse cost matrix
edges = []
costs = []
for i in range(n):
    for j in indices[i]:
        edges.append((i, j))
        costs.append(distances[i, np.where(indices[i] == j)[0][0]])

Ws = sparse.COO(edges, costs, shape=(n, n), fill_value=1e9)

# Run Sinkhorn
f, err, h = sinkhorn_w1_sparse(Ws, z, epsilon=0.01, niter=1000)
```

### 3D Mesh Graph

```python
# Build adjacency from mesh faces
A = np.zeros((n_vertices, n_vertices))
for face in faces:
    for i in range(3):
        v1, v2 = face[i], face[(i+1) % 3]
        A[v1, v2] = 1
        A[v2, v1] = 1

# Edge distances
W = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2)
W[A == 0] = 1e9

# Run Sinkhorn
f, err, h = sinkhorn_w1(W, z, epsilon=0.1, niter=2000)
```

## 7. Troubleshooting

**ImportError: No module named 'sparse'**
```bash
pip install sparse
```

**ImportError: No module named 'cvxpy'**
```bash
pip install cvxpy
```

**Slow convergence**
- Increase epsilon for faster convergence (less accurate)
- Increase niter
- Use sparse version for large graphs

**Numerical overflow**
- The algorithm uses stabilized log-sum-exp
- Should not happen with reasonable epsilon (> 1e-3)
- Check that cost matrix W is reasonable

## 8. Next Steps

- Read the [full documentation](README.md)
- Check out the [examples](examples/)
- Read the [paper](paper/flow-sinkhorn.tex)
- [Contribute](CONTRIBUTING.md) to the project

## Questions?

Open an issue on GitHub: https://github.com/gpeyre/flow-sinkhorn/issues
