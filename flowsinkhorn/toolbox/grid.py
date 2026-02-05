"""
Grid-graph helpers used by the obstacle-transport examples.
"""

import numpy as np


def create_grid_graph(grid_size):
    """
    Create a 2D square grid graph.

    Parameters
    ----------
    grid_size : int
        Number of rows/columns in the square grid.

    Returns
    -------
    positions : ndarray of shape (grid_size*grid_size, 2)
        Vertex coordinates as (x, y).
    adjacency : ndarray of shape (n, n)
        Symmetric adjacency matrix.
    edges : list[tuple[int, int]]
        Undirected edge list (i, j), with i < j.
    """
    n = grid_size * grid_size
    positions = np.zeros((n, 2), dtype=float)
    adjacency = np.zeros((n, n), dtype=int)
    edges = []

    for row in range(grid_size):
        for col in range(grid_size):
            idx = row * grid_size + col
            positions[idx] = (col, row)

            if col < grid_size - 1:
                right = row * grid_size + (col + 1)
                adjacency[idx, right] = 1
                adjacency[right, idx] = 1
                edges.append((idx, right))

            if row < grid_size - 1:
                down = (row + 1) * grid_size + col
                adjacency[idx, down] = 1
                adjacency[down, idx] = 1
                edges.append((idx, down))

    return positions, adjacency, edges


def gaussian_bump(points, center, sigma):
    """
    Evaluate a Gaussian bump on 2D points.

    Parameters
    ----------
    points : ndarray of shape (n, 2)
        Point coordinates.
    center : tuple[float, float]
        Gaussian center.
    sigma : float
        Standard deviation.

    Returns
    -------
    ndarray of shape (n,)
        Bump values at each point.
    """
    center = np.asarray(center, dtype=float)
    sq_dist = np.sum((points - center[None, :]) ** 2, axis=1)
    return np.exp(-sq_dist / (2 * sigma**2))


def compute_modulated_costs(
    positions,
    adjacency,
    edges,
    centers,
    sigmas,
    alpha=10.0,
    non_edge_cost=1e9,
):
    """
    Build a cost matrix by modulating edge lengths with Gaussian obstacles.

    Parameters
    ----------
    positions : ndarray of shape (n, 2)
        Vertex coordinates.
    adjacency : ndarray of shape (n, n)
        Graph adjacency matrix.
    edges : list[tuple[int, int]]
        Edge list.
    centers : list[tuple[float, float]]
        Centers of Gaussian bumps.
    sigmas : list[float]
        Standard deviations of each bump.
    alpha : float, default=10.0
        Obstacle amplitude.
    non_edge_cost : float, default=1e9
        Cost assigned to non-edges.

    Returns
    -------
    cost_matrix : ndarray of shape (n, n)
        Symmetric edge-cost matrix.
    cost_field : ndarray of shape (n,)
        Per-vertex obstacle field (for visualization).
    """
    if len(centers) != len(sigmas):
        raise ValueError("centers and sigmas must have the same length.")

    n = len(positions)
    cost_field = np.zeros(n, dtype=float)
    for center, sigma in zip(centers, sigmas):
        cost_field += gaussian_bump(positions, center, sigma)

    cost_matrix = np.zeros((n, n), dtype=float)
    for i, j in edges:
        midpoint = (positions[i] + positions[j]) / 2
        bump_value = 0.0
        for center, sigma in zip(centers, sigmas):
            bump_value += gaussian_bump(midpoint[None, :], center, sigma)[0]

        base = np.linalg.norm(positions[i] - positions[j])
        edge_cost = base * (1 + alpha * bump_value)
        cost_matrix[i, j] = edge_cost
        cost_matrix[j, i] = edge_cost

    cost_matrix[adjacency == 0] = non_edge_cost
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix, cost_field


def create_corner_sources_sinks(grid_size):
    """
    Create one source in top-left and one sink in bottom-right.

    Parameters
    ----------
    grid_size : int
        Grid size.

    Returns
    -------
    source_idx : int
        Source index.
    sink_idx : int
        Sink index.
    z : ndarray of shape (grid_size*grid_size,)
        Source/sink vector.
    """
    n = grid_size * grid_size
    source_idx = 0
    sink_idx = n - 1
    z = np.zeros(n, dtype=float)
    z[source_idx] = 1.0
    z[sink_idx] = -1.0
    return source_idx, sink_idx, z


def plot_grid_with_costs(positions, cost_field, grid_size, title="Grid with Cost Field"):
    """
    Plot grid obstacle field as a heatmap.
    """
    import matplotlib.pyplot as plt

    _ = positions  # kept for API symmetry with notebook code
    cost_grid = cost_field.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(
        cost_grid,
        cmap="YlOrRd",
        origin="upper",
        extent=[0, grid_size, grid_size, 0],
        alpha=0.8,
    )
    plt.colorbar(im, ax=ax, label="Cost Field")

    for i in range(0, grid_size, 3):
        ax.axhline(y=i, color="gray", linewidth=0.3, alpha=0.3)
        ax.axvline(x=i, color="gray", linewidth=0.3, alpha=0.3)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_grid_with_flow(
    positions,
    cost_field,
    flow_matrix,
    z,
    grid_size,
    threshold=1e-6,
    title="Grid with Flow",
    flow_color="purple",
    flow_width_scale=5,
):
    """
    Plot flow edges on top of the grid heatmap.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    cost_grid = cost_field.reshape(grid_size, grid_size)
    fig, ax = plt.subplots(figsize=(14, 13))
    im = ax.imshow(
        cost_grid,
        cmap="YlOrRd",
        origin="upper",
        extent=[0, grid_size, grid_size, 0],
        alpha=0.6,
    )
    plt.colorbar(im, ax=ax, label="Cost Field")

    segments = []
    weights = []
    max_flow = flow_matrix.max()
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            flow_ij = max(flow_matrix[i, j], flow_matrix[j, i])
            if flow_ij > threshold:
                segments.append([positions[i], positions[j]])
                weights.append(flow_ij)

    if segments:
        weights = np.asarray(weights)
        normalized = weights / max_flow
        lc = LineCollection(
            segments,
            linewidths=normalized * flow_width_scale,
            colors=flow_color,
            alpha=0.8,
            zorder=5,
        )
        ax.add_collection(lc)

    sources = np.where(z > 0)[0]
    sinks = np.where(z < 0)[0]
    if len(sources) > 0:
        ax.scatter(
            positions[sources, 0],
            positions[sources, 1],
            s=400,
            c="blue",
            marker="s",
            edgecolors="black",
            linewidths=3,
            label="Source",
            zorder=10,
        )
    if len(sinks) > 0:
        ax.scatter(
            positions[sinks, 0],
            positions[sinks, 1],
            s=400,
            c="green",
            marker="s",
            edgecolors="black",
            linewidths=3,
            label="Sink",
            zorder=10,
        )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(
        f"{title}\n({len(segments)} edges with flow > {threshold:.2e})",
        fontsize=14,
    )
    ax.legend(fontsize=12, loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
