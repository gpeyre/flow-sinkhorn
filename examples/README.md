# Flow Sinkhorn Examples

This directory contains example notebooks demonstrating the Flow Sinkhorn algorithm.

## Available Examples

### `planar-graph.ipynb`

Main demonstration notebook showcasing:

- **Graph generation**: Creating random planar K-NN graphs
- **Exact solver**: Computing optimal transport using linear programming (CVXPY)
- **Sinkhorn solvers**:
  - Dense implementation for small/dense graphs
  - Sparse implementation for large sparse graphs
- **Convergence analysis**: Monitoring error evolution over iterations
- **Regularization study**: Effect of the epsilon parameter
- **Visualization**: Flow patterns on graphs

### `mesh-transport.ipynb`

Advanced 3D mesh example demonstrating:

- **3D mesh loading**: Reading OFF format files
- **Graph construction**: Building adjacency from mesh edges
- **3D visualization**: Interactive mesh plots with highlighted vertices
- **Source/sink selection**: Automatic selection based on geometry (top/bottom vertices)
- **Exact transport**: Computing optimal flow using CVXPY
- **Flow Sinkhorn**: Comparing results with different regularization levels:
  - Large ε for faster, more diffuse flows
  - Small ε for accurate, sparse flows
- **Flow visualization**: Displaying transport paths on 3D mesh
- **Performance comparison**: Timing and accuracy analysis

This example uses the `data/moomoo.off` mesh file and showcases optimal transport on real 3D geometry.

### `grid-transport.ipynb`

2D grid with obstacles example demonstrating:

- **Grid construction**: Creating regular 30×30 square grid graphs
- **Cost modulation**: Using Gaussian bumps to create high-cost obstacle regions
- **Obstacle avoidance**: Visualizing how optimal transport routes around obstacles
- **Corner-to-corner transport**: Source at top-left, sink at bottom-right
- **Exact transport**: Computing optimal paths using CVXPY
- **Flow Sinkhorn**: Comparing results with different regularization levels:
  - Large ε for diffuse, multi-path flows
  - Small ε for sparse, concentrated paths
- **2D flow visualization**: Line thickness proportional to flow intensity
- **Heatmap overlays**: Cost field visualization with flow paths

This example illustrates **obstacle avoidance** behavior and is ideal for understanding path planning and routing applications.

### `gpu-benchmark.ipynb`

GPU acceleration benchmark demonstrating:

- **Device detection**: Automatic GPU/CPU detection (CUDA, MPS, or CPU)
- **PyTorch implementation**: GPU-accelerated Flow Sinkhorn using PyTorch
- **Numerical equivalence**: Verification that PyTorch matches NumPy to machine precision
- **Performance benchmarking**: Wall-clock time comparison for fixed iterations
- **Scaling tests**: Performance vs graph size analysis
- **Speedup measurements**: Quantifying GPU acceleration (10-100x typical)
- **Device flexibility**: Testing on CPU, CUDA (NVIDIA), and MPS (Apple Silicon)

This example validates the PyTorch implementation and demonstrates the performance benefits of GPU acceleration for large-scale optimal transport problems.

## Running the Notebooks

### Local Installation

1. Install the package with example dependencies:
   ```bash
   cd ..
   pip install -e ".[examples]"
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook planar-graph.ipynb
   ```

For GPU acceleration (gpu-benchmark.ipynb):
   ```bash
   pip install -e ".[gpu]"  # Install PyTorch
   jupyter notebook gpu-benchmark.ipynb
   ```

### Google Colab

Click the badge in the main README to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpeyre/flow-sinkhorn/blob/main/examples/planar-graph.ipynb)

## Dependencies

The examples require:
- `numpy` - Numerical computations
- `scipy` - Special functions (log-sum-exp)
- `matplotlib` - Plotting
- `networkx` - Graph visualization
- `scikit-learn` - K-NN graph construction
- `pandas` - Data tables (for benchmarks)
- `sparse` (optional) - Sparse matrix operations
- `cvxpy` (optional) - Exact linear programming solver
- `torch` (optional) - GPU acceleration via PyTorch

Install all at once with:
```bash
pip install -e ".[all]"
```

For GPU support only:
```bash
pip install -e ".[gpu]"
```
