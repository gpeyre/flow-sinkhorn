"""
Mesh/3D helpers used by mesh-transport examples.
"""

import numpy as np


def load_off_file(filename):
    """
    Load a 3D mesh from an OFF file.

    Parameters
    ----------
    filename : str
        Path to OFF file.

    Returns
    -------
    vertices : ndarray of shape (n_vertices, 3)
        Vertex coordinates.
    faces : ndarray of shape (n_faces, 3)
        Triangle face indices (non-triangle faces are truncated to first 3).
    """
    with open(filename, "r") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError("Not a valid OFF file")

        n_vertices, n_faces, _ = map(int, f.readline().strip().split())

        vertices = np.zeros((n_vertices, 3), dtype=float)
        for i in range(n_vertices):
            vertices[i] = [float(x) for x in f.readline().strip().split()[:3]]

        faces = []
        for _ in range(n_faces):
            parts = f.readline().strip().split()
            n_verts = int(parts[0])
            faces.append([int(x) for x in parts[1 : n_verts + 1]][:3])

    return vertices, np.asarray(faces, dtype=int)


def build_mesh_graph(vertices, faces, non_edge_cost=1e9):
    """
    Build an undirected graph from triangle mesh connectivity.

    Parameters
    ----------
    vertices : ndarray of shape (n, 3)
        Vertex coordinates.
    faces : ndarray of shape (m, 3)
        Triangle face indices.
    non_edge_cost : float, default=1e9
        Cost assigned to non-edges.

    Returns
    -------
    adjacency : ndarray of shape (n, n)
        Binary adjacency matrix.
    cost_matrix : ndarray of shape (n, n)
        Euclidean edge-distance matrix.
    edges : list[tuple[int, int]]
        Undirected edge list.
    """
    n = len(vertices)
    adjacency = np.zeros((n, n), dtype=int)
    edge_set = set()

    for face in faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            edge_set.add(tuple(sorted((a, b))))

    edges = list(edge_set)
    for i, j in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    cost_matrix = np.zeros((n, n), dtype=float)
    for i, j in edges:
        d = np.linalg.norm(vertices[i] - vertices[j])
        cost_matrix[i, j] = d
        cost_matrix[j, i] = d

    cost_matrix[adjacency == 0] = non_edge_cost
    np.fill_diagonal(cost_matrix, 0)
    return adjacency, cost_matrix, edges


def select_sources_sinks(vertices, k=3, axis=2):
    """
    Select source/sink vertices from extrema on a chosen axis.

    Parameters
    ----------
    vertices : ndarray of shape (n, 3)
        Vertex coordinates.
    k : int, default=3
        Number of sources and sinks.
    axis : int, default=2
        Coordinate axis (0=x, 1=y, 2=z).

    Returns
    -------
    sources : ndarray
        Source indices.
    sinks : ndarray
        Sink indices.
    z : ndarray of shape (n,)
        Normalized source/sink vector.
    """
    n = len(vertices)
    coords = vertices[:, axis]
    sources = np.argsort(coords)[-k:]
    sinks = np.argsort(coords)[:k]

    z = np.zeros(n, dtype=float)
    z[sources] = 1.0 / k
    z[sinks] = -1.0 / k
    return sources, sinks, z


def plot_mesh(
    vertices,
    faces,
    title="3D Mesh",
    highlight_vertices=None,
    highlight_colors=None,
    vertex_size=50,
    elev=20,
    azim=45,
    show_edges=True,
    alpha=0.8,
    figsize=(12, 10),
):
    """
    Plot mesh surface with optional highlighted vertices.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    mesh = [vertices[face] for face in faces]
    poly = Poly3DCollection(
        mesh,
        alpha=alpha,
        facecolors="lightgray",
        linewidths=0.3 if show_edges else 0,
        edgecolors="black" if show_edges else None,
    )
    ax.add_collection3d(poly)

    if highlight_vertices is not None:
        for label, indices in highlight_vertices.items():
            color = highlight_colors.get(label, "red") if highlight_colors else "red"
            ax.scatter(
                vertices[indices, 0],
                vertices[indices, 1],
                vertices[indices, 2],
                c=color,
                s=vertex_size,
                label=label,
                alpha=1.0,
                edgecolors="black",
                linewidths=2,
                zorder=10,
            )
        ax.legend(fontsize=11)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.set_title(title, fontsize=14, pad=20)
    ax.view_init(elev=elev, azim=azim)

    max_range = np.array(
        [
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min(),
        ]
    ).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig, ax


def plot_mesh_with_flow(
    vertices,
    faces,
    flow_matrix,
    z,
    threshold=1e-6,
    title="Mesh with Flow",
    flow_color="purple",
    flow_width_scale=3,
    elev=20,
    azim=45,
    surface_alpha=0.3,
    figsize=(14, 12),
):
    """
    Plot mesh surface with transport flow overlaid on active edges.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    mesh = [vertices[face] for face in faces]
    poly = Poly3DCollection(
        mesh, alpha=surface_alpha, facecolors="lightgray", linewidths=0.1, edgecolors="gray"
    )
    ax.add_collection3d(poly)

    flow_edges = []
    flow_weights = []
    max_flow = flow_matrix.max()
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            flow_ij = max(flow_matrix[i, j], flow_matrix[j, i])
            if flow_ij > threshold:
                flow_edges.append([vertices[i], vertices[j]])
                flow_weights.append(flow_ij)

    if flow_edges:
        flow_weights = np.asarray(flow_weights)
        normalized = flow_weights / max_flow
        for edge, weight in zip(flow_edges, normalized):
            edge = np.asarray(edge)
            ax.plot(
                edge[:, 0],
                edge[:, 1],
                edge[:, 2],
                color=flow_color,
                linewidth=weight * flow_width_scale + 1,
                alpha=0.9,
                zorder=5,
            )

    sources = np.where(z > 0)[0]
    sinks = np.where(z < 0)[0]
    if len(sources) > 0:
        ax.scatter(
            vertices[sources, 0],
            vertices[sources, 1],
            vertices[sources, 2],
            c="red",
            s=200,
            label="Sources",
            alpha=1.0,
            edgecolors="black",
            linewidths=3,
            zorder=10,
        )
    if len(sinks) > 0:
        ax.scatter(
            vertices[sinks, 0],
            vertices[sinks, 1],
            vertices[sinks, 2],
            c="blue",
            s=200,
            label="Sinks",
            alpha=1.0,
            edgecolors="black",
            linewidths=3,
            zorder=10,
        )

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    ax.set_title(f"{title}\n({len(flow_edges)} edges with flow > {threshold:.2e})", fontsize=14, pad=20)
    ax.legend(fontsize=12, loc="upper left")
    ax.view_init(elev=elev, azim=azim)

    max_range = np.array(
        [
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min(),
        ]
    ).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    return fig, ax
