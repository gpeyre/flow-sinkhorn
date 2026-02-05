"""
Planar/random graph helpers used by benchmark notebooks.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def create_planar_knn_graph(n, k=5, seed=42, non_edge_cost=1e9):
    """
    Create a random 2D K-NN graph and its transport cost matrix.

    Parameters
    ----------
    n : int
        Number of vertices.
    k : int, default=5
        Number of nearest neighbors used to build graph edges.
    seed : int, default=42
        Random seed.
    non_edge_cost : float, default=1e9
        Cost assigned to non-edges in the dense cost matrix.

    Returns
    -------
    positions : ndarray of shape (2, n)
        Random vertex coordinates in [0, 1]^2.
    adjacency : ndarray of shape (n, n)
        Symmetric adjacency matrix.
    cost_matrix : ndarray of shape (n, n)
        Dense cost matrix where non-edges have `non_edge_cost`.
    """
    rng = np.random.default_rng(seed)
    positions = rng.random((2, n))

    nbrs = NearestNeighbors(n_neighbors=k).fit(positions.T)
    _, indices = nbrs.kneighbors(positions.T)

    adjacency = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in indices[i]:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
    np.fill_diagonal(adjacency, 0)

    cost_matrix = np.where(adjacency > 0, 1.0, non_edge_cost).astype(float)
    np.fill_diagonal(cost_matrix, 0)
    return positions, adjacency, cost_matrix


def create_source_sink_from_positions(positions, adjacency, diffusion_steps=1):
    """
    Create a normalized source/sink vector from geometric extrema.

    The source is initialized near the top-left diagonal extreme (min x+y),
    and the sink near the opposite extreme (max x+y). Optional diffusion on
    the graph adjacency broadens support before normalization.

    Parameters
    ----------
    positions : ndarray of shape (2, n)
        Vertex coordinates.
    adjacency : ndarray of shape (n, n)
        Graph adjacency matrix used for diffusion.
    diffusion_steps : int, default=1
        Number of linear diffusion steps.

    Returns
    -------
    z : ndarray of shape (n,)
        Normalized source/sink vector with sum(z)=0.
    """
    n = positions.shape[1]
    z = np.zeros(n, dtype=float)
    z[np.argmin(positions[0] + positions[1])] = 1.0
    z[np.argmax(positions[0] + positions[1])] = -1.0

    for _ in range(diffusion_steps):
        z = adjacency @ z + z

    z = np.sign(z)
    pos = z > 0
    neg = z < 0
    if np.any(pos):
        z[pos] = z[pos] / np.sum(z[pos])
    if np.any(neg):
        z[neg] = -z[neg] / np.sum(z[neg])
    return z
