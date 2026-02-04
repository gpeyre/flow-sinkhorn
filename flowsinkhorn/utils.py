"""
Utility functions for graph manipulation and visualization.

This module provides helper functions for:
- Loading and processing 3D meshes
- Building graph structures from meshes
- Selecting sources and sinks
- Visualizing graphs and flows

References
----------
Gabriel Peyré, "Robust Sublinear Convergence Rates for Iterative Bregman
Projections on Affine Spaces", arXiv preprint, 2026.
"""

import numpy as np


def load_off_file(filename):
    """
    Load a 3D mesh from an OFF file.

    The OFF (Object File Format) is a simple format for storing 3D geometry:
    - First line: "OFF"
    - Second line: number of vertices, number of faces, number of edges
    - Next lines: vertex coordinates (x, y, z)
    - Remaining lines: face vertex indices

    Parameters
    ----------
    filename : str
        Path to the OFF file

    Returns
    -------
    vertices : ndarray of shape (n_vertices, 3)
        Vertex coordinates
    faces : ndarray of shape (n_faces, 3)
        Face vertex indices (assuming triangular mesh)

    Examples
    --------
    >>> vertices, faces = load_off_file('mesh.off')
    >>> print(f"Loaded {len(vertices)} vertices, {len(faces)} faces")
    """
    with open(filename, 'r') as f:
        # Read header
        line = f.readline().strip()
        if line != 'OFF':
            raise ValueError("Not a valid OFF file")

        # Read counts
        line = f.readline().strip()
        n_vertices, n_faces, n_edges = map(int, line.split())

        # Read vertices
        vertices = np.zeros((n_vertices, 3))
        for i in range(n_vertices):
            line = f.readline().strip().split()
            vertices[i] = [float(x) for x in line[:3]]

        # Read faces
        faces = []
        for i in range(n_faces):
            line = f.readline().strip().split()
            n_verts = int(line[0])
            face_verts = [int(x) for x in line[1:n_verts+1]]
            # If not triangular, take first 3 vertices
            faces.append(face_verts[:3])

        faces = np.array(faces)

    return vertices, faces


def build_mesh_graph(vertices, faces):
    """
    Build adjacency matrix and distance matrix from mesh.

    Parameters
    ----------
    vertices : ndarray of shape (n, 3)
        Vertex coordinates
    faces : ndarray of shape (m, 3)
        Face vertex indices

    Returns
    -------
    A : ndarray of shape (n, n)
        Adjacency matrix (binary)
    W : ndarray of shape (n, n)
        Distance matrix (Euclidean distances on edges, large for non-edges)
    edges : list of tuples
        List of (i, j) edge pairs

    Examples
    --------
    >>> A, W, edges = build_mesh_graph(vertices, faces)
    >>> print(f"Graph has {len(edges)} edges")
    """
    n = len(vertices)
    A = np.zeros((n, n))

    # Extract edges from faces
    edges_set = set()
    for face in faces:
        # Add all three edges of the triangle
        for i in range(3):
            v1, v2 = face[i], face[(i+1) % 3]
            # Store as sorted tuple for undirected graph
            edge = tuple(sorted([v1, v2]))
            edges_set.add(edge)

    edges = list(edges_set)

    # Build adjacency matrix
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    # Compute edge distances
    W = np.zeros((n, n))
    for i, j in edges:
        dist = np.linalg.norm(vertices[i] - vertices[j])
        W[i, j] = dist
        W[j, i] = dist

    # Set large value for non-edges
    W[A == 0] = 1e9
    np.fill_diagonal(W, 0)

    return A, W, edges


def select_sources_sinks(vertices, k=3, axis=2):
    """
    Select source and sink vertices based on coordinate along an axis.

    Parameters
    ----------
    vertices : ndarray of shape (n, 3)
        Vertex coordinates
    k : int, default=3
        Number of sources and sinks
    axis : int, default=2
        Axis to use (0=x, 1=y, 2=z)

    Returns
    -------
    sources : ndarray
        Indices of source vertices
    sinks : ndarray
        Indices of sink vertices
    z : ndarray of shape (n,)
        Source/sink vector

    Examples
    --------
    >>> sources, sinks, z = select_sources_sinks(vertices, k=3)
    >>> print(f"Sources at z={vertices[sources, 2]}")
    """
    n = len(vertices)
    coords = vertices[:, axis]

    # Select top k vertices as sources
    sources = np.argsort(coords)[-k:]

    # Select bottom k vertices as sinks
    sinks = np.argsort(coords)[:k]

    # Create source/sink vector
    z = np.zeros(n)
    z[sources] = 1.0 / k  # Uniform distribution on sources
    z[sinks] = -1.0 / k   # Uniform distribution on sinks

    return sources, sinks, z


def plot_mesh(vertices, faces, title="3D Mesh", highlight_vertices=None,
              highlight_colors=None, vertex_size=50, elev=20, azim=45,
              show_edges=True, alpha=0.8, figsize=(12, 10)):
    """
    Plot a 3D mesh as a shaded surface with optional highlighted vertices.

    Parameters
    ----------
    vertices : ndarray of shape (n, 3)
        Vertex coordinates
    faces : ndarray of shape (m, 3)
        Face indices
    title : str
        Plot title
    highlight_vertices : dict, optional
        Dict mapping labels to lists of vertex indices to highlight
    highlight_colors : dict, optional
        Dict mapping labels to colors
    vertex_size : int
        Size of highlighted vertices
    elev, azim : float
        Viewing angle
    show_edges : bool
        Show mesh edges as black lines
    alpha : float
        Surface transparency (0=transparent, 1=opaque)
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes

    Examples
    --------
    >>> fig, ax = plot_mesh(vertices, faces,
    ...                      highlight_vertices={'Sources': [0, 1], 'Sinks': [10, 11]},
    ...                      highlight_colors={'Sources': 'red', 'Sinks': 'blue'})
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh surface
    mesh = []
    for face in faces:
        triangle = vertices[face]
        mesh.append(triangle)

    # Plot shaded surface
    poly = Poly3DCollection(mesh, alpha=alpha, facecolors='lightgray',
                            linewidths=0.3 if show_edges else 0,
                            edgecolors='black' if show_edges else None)
    ax.add_collection3d(poly)

    # Highlight specific vertices
    if highlight_vertices is not None:
        for label, indices in highlight_vertices.items():
            color = highlight_colors.get(label, 'red') if highlight_colors else 'red'
            ax.scatter(vertices[indices, 0], vertices[indices, 1], vertices[indices, 2],
                      c=color, s=vertex_size, label=label, alpha=1.0,
                      edgecolors='black', linewidths=2, zorder=10)
        ax.legend(fontsize=11)

    # Set limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    # Remove grid and axes for cleaner look
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    ax.set_title(title, fontsize=14, pad=20)
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    return fig, ax


def plot_mesh_with_flow(vertices, faces, F, z, threshold=1e-6,
                        title="Mesh with Flow", flow_color='purple',
                        flow_width_scale=3, elev=20, azim=45,
                        surface_alpha=0.3, figsize=(14, 12)):
    """
    Plot mesh with flow edges highlighted as thick colored lines.

    The mesh is rendered as a transparent shaded surface, with flow edges
    drawn as thick lines proportional to flow intensity.

    Parameters
    ----------
    vertices : ndarray of shape (n, 3)
        Vertex coordinates
    faces : ndarray of shape (m, 3)
        Face indices
    F : ndarray of shape (n, n)
        Flow matrix
    z : ndarray of shape (n,)
        Source/sink vector
    threshold : float
        Minimum flow to display
    title : str
        Plot title
    flow_color : str or tuple
        Color for flow edges
    flow_width_scale : float
        Scale for flow line width
    elev, azim : float
        Viewing angle
    surface_alpha : float
        Surface transparency (lower = more transparent)
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes

    Examples
    --------
    >>> fig, ax = plot_mesh_with_flow(vertices, faces, F, z,
    ...                                threshold=1e-5, flow_color='purple')
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Create mesh surface (very transparent)
    mesh = []
    for face in faces:
        triangle = vertices[face]
        mesh.append(triangle)

    poly = Poly3DCollection(mesh, alpha=surface_alpha, facecolors='lightgray',
                            linewidths=0.1, edgecolors='gray')
    ax.add_collection3d(poly)

    # Extract and plot flow edges
    flow_edges = []
    flow_weights = []
    max_flow = F.max()

    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            flow_ij = max(F[i, j], F[j, i])
            if flow_ij > threshold:
                flow_edges.append([vertices[i], vertices[j]])
                flow_weights.append(flow_ij)

    # Plot flow edges as thick lines
    if len(flow_edges) > 0:
        flow_weights = np.array(flow_weights)
        normalized_weights = flow_weights / max_flow

        for edge, weight in zip(flow_edges, normalized_weights):
            edge = np.array(edge)
            # Use thicker lines for better visibility
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2],
                   color=flow_color, linewidth=weight*flow_width_scale + 1,
                   alpha=0.9, zorder=5)

    # Plot sources and sinks
    sources = np.where(z > 0)[0]
    sinks = np.where(z < 0)[0]

    if len(sources) > 0:
        ax.scatter(vertices[sources, 0], vertices[sources, 1], vertices[sources, 2],
                  c='red', s=200, label='Sources', alpha=1.0,
                  edgecolors='black', linewidths=3, zorder=10)
    if len(sinks) > 0:
        ax.scatter(vertices[sinks, 0], vertices[sinks, 1], vertices[sinks, 2],
                  c='blue', s=200, label='Sinks', alpha=1.0,
                  edgecolors='black', linewidths=3, zorder=10)

    # Set limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    # Remove grid and axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Make panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    ax.set_title(f"{title}\\n({len(flow_edges)} edges with flow > {threshold:.2e})",
                fontsize=14, pad=20)
    ax.legend(fontsize=12, loc='upper left')
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    return fig, ax
