"""
Flow-based Sinkhorn algorithm for W1 optimal transport on graphs.

This module implements the Sinkhorn-flow algorithm for approximating the
Wasserstein-1 distance on a graph using entropic regularization.

References
----------
Gabriel Peyré, "Robust Sublinear Convergence Rates for Iterative Bregman
Projections on Affine Spaces", arXiv preprint, 2026.
"""

import numpy as np
import scipy as sp
try:
    import sparse
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False


def _lse(Z, u):
    """
    Stabilized log-sum-exp operation along axis 1.

    Computes log(sum_j exp(Z[i,j] + u[j])) in a numerically stable way.

    Parameters
    ----------
    Z : ndarray of shape (n, n)
        Input matrix.
    u : ndarray of shape (n,)
        Vector to be added before exponential.

    Returns
    -------
    ndarray of shape (n,)
        Log-sum-exp result.
    """
    return sp.special.logsumexp(Z + u[None, :], axis=1)


def _lse_sparse(Z, u):
    """
    Stabilized sparse log-sum-exp operation along axis 1.

    Computes log(sum_j exp(Z[i,j] + u[j])) for sparse Z in a numerically
    stable way.

    Parameters
    ----------
    Z : sparse.COO matrix of shape (n, n)
        Sparse input matrix.
    u : ndarray of shape (n,)
        Vector to be added before exponential.

    Returns
    -------
    ndarray of shape (n,)
        Log-sum-exp result.
    """
    if not SPARSE_AVAILABLE:
        raise ImportError("The 'sparse' package is required for sparse operations. "
                          "Install it with: pip install sparse")

    Z1 = Z.copy()
    # Add u[j] to each element Z[i,j]
    Z1.data = Z1.data + u[Z1.coords[1]]
    # Stabilization: subtract row-wise maximum
    m = np.max(Z1, axis=1).todense()
    Z1.data = Z1.data - m[Z1.coords[0]]
    return np.log(np.sum(np.exp(Z1), axis=1)).todense() + m


def sinkhorn_w1(W, z, epsilon, niter):
    """
    W1-Sinkhorn algorithm for entropic regularized optimal transport on graphs.

    This function solves the entropic regularized W1 optimal transport problem:

        min_{f >= 0} <f, W> + epsilon * KL(f | 1)
        s.t. f^T 1 - f 1 = z

    where f is the flow matrix, W is the cost matrix, and z is the
    source/sink vector.

    The algorithm iteratively computes:
        f[i,j] = exp((-W[i,j] + h[i] - h[j]) / epsilon)

    with the update:
        h <- h/2 - (epsilon/2) * m

    where m is computed from the flow constraints.

    Parameters
    ----------
    W : ndarray of shape (n, n)
        Cost/distance matrix between nodes. W[i,j] represents the cost of
        moving mass from node j to node i. For non-edges, W should be large.
    z : ndarray of shape (n,)
        Source/sink vector. Positive values are sources, negative are sinks,
        and sum(z) should be 0.
    epsilon : float
        Entropic regularization parameter. Smaller values lead to solutions
        closer to the unregularized problem but may require more iterations.
    niter : int
        Number of iterations to perform.

    Returns
    -------
    f : ndarray of shape (n, n)
        Flow matrix. f[i,j] is the flow from node j to node i.
    err : list of float
        L1 error at each iteration: ||f^T 1 - f 1 - z||_1
    h : ndarray of shape (n,)
        Dual variable (potential) at each node.

    Notes
    -----
    The algorithm distinguishes three cases for numerical stability:
    - Nodes with z[i] = 0 (neutral nodes)
    - Nodes with z[i] > 0 (sources)
    - Nodes with z[i] < 0 (sinks)

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple graph with adjacency matrix A
    >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> W = 1 / (A + 1e-9)  # Cost matrix (large for non-edges)
    >>> z = np.array([1.0, 0.0, -1.0])  # Source at 0, sink at 2
    >>> f, err, h = sinkhorn_w1(W, z, epsilon=0.1, niter=100)
    """
    n = len(z)
    K = np.exp(-W / epsilon)
    h = np.zeros(n)
    r = z / 2
    err = []

    # Identify node types for numerical stability
    I0 = np.where(z == 0)[0]  # Neutral nodes
    Ip = np.where(z > 0)[0]   # Source nodes
    In = np.where(z < 0)[0]   # Sink nodes

    for it in range(niter):
        # Compute a[i] = sum_j exp((-W[i,j] - h[i]) / epsilon)
        # and b[i] = sum_j exp((-W[i,j] + h[i]) / epsilon)
        a = K @ np.exp(-h / epsilon)
        b = K @ np.exp(+h / epsilon)

        # Log-domain versions for stability
        loga = _lse(-W / epsilon, -h / epsilon)
        logb = _lse(-W / epsilon, +h / epsilon)

        # Compute update vector m, treating each case separately
        m = np.zeros(n)
        m[I0] = (loga[I0] - logb[I0]) / 2
        m[Ip] = np.log(np.sqrt(r[Ip]**2 + a[Ip] * b[Ip]) + r[Ip]) - logb[Ip]
        m[In] = -np.log(np.sqrt(r[In]**2 + a[In] * b[In]) - r[In]) + loga[In]

        # Update potential
        h = h / 2 - epsilon / 2 * m

        # Compute flow from potentials
        f = np.exp((-W + h[:, None] - h[None, :]) / epsilon)

        # Compute constraint error
        e = np.linalg.norm((np.sum(f, axis=0) - np.sum(f, axis=1)) - z, 1)
        err.append(e)

    return f, err, h


def sinkhorn_w1_sparse(W, z, epsilon, niter):
    """
    Sparse W1-Sinkhorn algorithm for entropic regularized optimal transport.

    This is a memory-efficient version of sinkhorn_w1 for sparse cost matrices.
    It uses sparse matrix operations to reduce memory usage and computation time
    when the graph has few edges relative to n^2.

    Parameters
    ----------
    W : sparse.COO matrix of shape (n, n)
        Sparse cost/distance matrix. W[i,j] represents the cost of moving mass
        from node j to node i. Non-edges should have a large fill_value.
    z : ndarray of shape (n,)
        Source/sink vector. Positive values are sources, negative are sinks,
        and sum(z) should be 0.
    epsilon : float
        Entropic regularization parameter. Smaller values lead to solutions
        closer to the unregularized problem but may require more iterations.
    niter : int
        Number of iterations to perform.

    Returns
    -------
    f : sparse.COO matrix of shape (n, n)
        Sparse flow matrix. f[i,j] is the flow from node j to node i.
    err : list of float
        L1 error at each iteration: ||f^T 1 - f 1 - z||_1
    h : ndarray of shape (n,)
        Dual variable (potential) at each node.

    Raises
    ------
    ImportError
        If the 'sparse' package is not installed.

    Notes
    -----
    This function requires the PyData Sparse package:
        pip install sparse

    The sparse implementation is particularly beneficial when the graph has
    O(n) edges rather than O(n^2) edges, as is common for k-nearest neighbor
    graphs or grid graphs.

    Examples
    --------
    >>> import numpy as np
    >>> import sparse
    >>> # Create a sparse graph
    >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> coords = A.nonzero()
    >>> Ws = sparse.COO(coords, A[coords], fill_value=1e9)
    >>> z = np.array([1.0, 0.0, -1.0])
    >>> f, err, h = sinkhorn_w1_sparse(Ws, z, epsilon=0.1, niter=100)
    """
    if not SPARSE_AVAILABLE:
        raise ImportError("The 'sparse' package is required for sparse operations. "
                          "Install it with: pip install sparse")

    n = len(z)
    K = np.exp(-W / epsilon)
    K.fill_value = 0
    h = np.zeros(n)
    r = z / 2
    err = []

    # Identify node types for numerical stability
    I0 = np.where(z == 0)[0]  # Neutral nodes
    Ip = np.where(z > 0)[0]   # Source nodes
    In = np.where(z < 0)[0]   # Sink nodes

    for it in range(niter):
        # Compute a and b using sparse matrix-vector products
        a = K @ np.exp(-h / epsilon)
        b = K @ np.exp(+h / epsilon)

        # Log-domain versions using sparse log-sum-exp
        loga = _lse_sparse(-W / epsilon, -h / epsilon)
        logb = _lse_sparse(-W / epsilon, +h / epsilon)

        # Compute update vector m, treating each case separately
        m = np.zeros(n)
        m[I0] = (loga[I0] - logb[I0]) / 2
        m[Ip] = np.log(np.sqrt(r[Ip]**2 + a[Ip] * b[Ip]) + r[Ip]) - logb[Ip]
        m[In] = -np.log(np.sqrt(r[In]**2 + a[In] * b[In]) - r[In]) + loga[In]

        # Update potential
        h = h / 2 - epsilon / 2 * m

        # Compute sparse flow from potentials
        f = -W.copy()
        f.data = f.data + h[f.coords[0]] - h[f.coords[1]]
        f = np.exp(f / epsilon)

        # Compute constraint error
        e = np.linalg.norm((np.sum(f, axis=0) - np.sum(f, axis=1)) - z, 1)
        err.append(e)

    return f, err, h
