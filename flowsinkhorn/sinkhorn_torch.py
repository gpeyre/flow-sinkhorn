"""
PyTorch implementation of Flow Sinkhorn with GPU acceleration.

This module provides GPU-accelerated versions of the Flow Sinkhorn algorithm
using PyTorch. It supports automatic device selection (CPU/CUDA/MPS) and
provides significant speedup on compatible hardware.

References
----------
Gabriel Peyré, "Robust Sublinear Convergence Rates for Iterative Bregman
Projections on Affine Spaces", arXiv preprint, 2026.
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device(device=None):
    """
    Get the best available device for computation.

    Parameters
    ----------
    device : str or torch.device, optional
        Requested device ('cpu', 'cuda', 'mps', or torch.device).
        If None, automatically selects the best available device.

    Returns
    -------
    torch.device
        The device to use for computation.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU acceleration. "
                          "Install it with: pip install torch")

    if device is not None:
        return torch.device(device)

    # Auto-select best device
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def _lse_torch(Z, u):
    """
    Stabilized log-sum-exp operation using PyTorch.

    Parameters
    ----------
    Z : torch.Tensor of shape (n, n)
        Input matrix.
    u : torch.Tensor of shape (n,)
        Vector to be added before exponential.

    Returns
    -------
    torch.Tensor of shape (n,)
        Log-sum-exp result.
    """
    return torch.logsumexp(Z + u[None, :], dim=1)


def _segment_logsumexp_from_sorted(values, row_ptr, n):
    """
    Row-wise log-sum-exp on sorted COO edge values without densifying.

    Parameters
    ----------
    values : torch.Tensor of shape (nnz,)
        Per-edge values sorted by row index.
    row_ptr : torch.Tensor of shape (n+1,)
        CSR-style row pointers.
    n : int
        Number of rows.

    Returns
    -------
    torch.Tensor of shape (n,)
        Row-wise log-sum-exp.
    """
    out = torch.full((n,), -torch.inf, dtype=values.dtype, device=values.device)
    for i in range(n):
        start = int(row_ptr[i].item())
        end = int(row_ptr[i + 1].item())
        if end > start:
            out[i] = torch.logsumexp(values[start:end], dim=0)
    return out


def sinkhorn_w1_torch(W, z, epsilon, niter, device=None, return_numpy=True):
    """
    PyTorch W1-Sinkhorn algorithm with GPU acceleration.

    This is a GPU-accelerated version of sinkhorn_w1 using PyTorch.
    It provides identical results to the NumPy version but can be much
    faster on GPU hardware.

    Parameters
    ----------
    W : ndarray or torch.Tensor of shape (n, n)
        Cost/distance matrix between nodes. W[i,j] represents the cost of
        moving mass from node j to node i. For non-edges, W should be large.
    z : ndarray or torch.Tensor of shape (n,)
        Source/sink vector. Positive values are sources, negative are sinks,
        and sum(z) should be 0.
    epsilon : float
        Entropic regularization parameter. Smaller values lead to solutions
        closer to the unregularized problem but may require more iterations.
    niter : int
        Number of iterations to perform.
    device : str or torch.device, optional
        Device to use for computation ('cpu', 'cuda', 'mps', or torch.device).
        If None, automatically selects the best available device.
    return_numpy : bool, default=True
        If True, return results as NumPy arrays. If False, return torch.Tensors.

    Returns
    -------
    f : ndarray or torch.Tensor of shape (n, n)
        Flow matrix. f[i,j] is the flow from node j to node i.
    err : list of float
        L1 error at each iteration: ||f^T 1 - f 1 - z||_1
    h : ndarray or torch.Tensor of shape (n,)
        Dual variable (potential) at each node.

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Notes
    -----
    This function uses the same algorithm as sinkhorn_w1 but leverages
    GPU acceleration for matrix operations. Results should be identical
    to the NumPy version within numerical precision.

    For very large graphs (n > 10000), GPU acceleration can provide
    10-100x speedup depending on hardware.

    Examples
    --------
    >>> import numpy as np
    >>> from flowsinkhorn.sinkhorn_torch import sinkhorn_w1_torch
    >>> # Create a simple graph
    >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> W = 1 / (A + 1e-9)
    >>> z = np.array([1.0, 0.0, -1.0])
    >>> # Run on best available device (GPU if available)
    >>> f, err, h = sinkhorn_w1_torch(W, z, epsilon=0.1, niter=100)
    >>> # Use specific device
    >>> f_cpu, err_cpu, h_cpu = sinkhorn_w1_torch(W, z, epsilon=0.1, niter=100, device='cpu')
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU acceleration. "
                          "Install it with: pip install torch")

    # Get device
    dev = get_device(device)

    # Convert inputs to torch tensors
    if isinstance(W, np.ndarray):
        W_torch = torch.from_numpy(W).float().to(dev)
    else:
        W_torch = W.float().to(dev)

    if isinstance(z, np.ndarray):
        z_torch = torch.from_numpy(z).float().to(dev)
    else:
        z_torch = z.float().to(dev)

    n = len(z_torch)

    # Initialize
    K = torch.exp(-W_torch / epsilon)
    h = torch.zeros(n, dtype=torch.float32, device=dev)
    r = z_torch / 2
    err = []

    # Identify node types for numerical stability
    I0 = torch.where(z_torch == 0)[0]
    Ip = torch.where(z_torch > 0)[0]
    In = torch.where(z_torch < 0)[0]

    for it in range(niter):
        # Compute a and b
        a = K @ torch.exp(-h / epsilon)
        b = K @ torch.exp(+h / epsilon)

        # Log-domain versions
        loga = _lse_torch(-W_torch / epsilon, -h / epsilon)
        logb = _lse_torch(-W_torch / epsilon, +h / epsilon)

        # Compute update vector m
        m = torch.zeros(n, dtype=torch.float32, device=dev)
        if len(I0) > 0:
            m[I0] = (loga[I0] - logb[I0]) / 2
        if len(Ip) > 0:
            m[Ip] = torch.log(torch.sqrt(r[Ip]**2 + a[Ip] * b[Ip]) + r[Ip]) - logb[Ip]
        if len(In) > 0:
            m[In] = -torch.log(torch.sqrt(r[In]**2 + a[In] * b[In]) - r[In]) + loga[In]

        # Update potential
        h = h / 2 - epsilon / 2 * m

        # Compute flow
        f = torch.exp((-W_torch + h[:, None] - h[None, :]) / epsilon)

        # Compute constraint error
        e = torch.norm((torch.sum(f, dim=0) - torch.sum(f, dim=1)) - z_torch, p=1)
        err.append(e.item())

    # Return results
    if return_numpy:
        return f.cpu().numpy(), err, h.cpu().numpy()
    else:
        return f, err, h


def sinkhorn_w1_torch_sparse(W_indices, W_values, W_shape, z, epsilon, niter,
                             device=None, return_numpy=True):
    """
    Sparse PyTorch W1-Sinkhorn algorithm with GPU acceleration.

    This is a sparse GPU-accelerated version using PyTorch sparse tensors.
    It is more memory-efficient for large sparse graphs.

    Parameters
    ----------
    W_indices : ndarray of shape (2, nnz)
        Indices of non-zero elements in W (COO format).
    W_values : ndarray of shape (nnz,)
        Values of non-zero elements in W.
    W_shape : tuple (n, n)
        Shape of the cost matrix.
    z : ndarray or torch.Tensor of shape (n,)
        Source/sink vector.
    epsilon : float
        Entropic regularization parameter.
    niter : int
        Number of iterations.
    device : str or torch.device, optional
        Device to use for computation.
    return_numpy : bool, default=True
        If True, return results as NumPy arrays.

    Returns
    -------
    f_indices : ndarray of shape (2, nnz)
        Indices of non-zero flow elements.
    f_values : ndarray
        Values of non-zero flow elements.
    err : list of float
        L1 error at each iteration.
    h : ndarray or torch.Tensor of shape (n,)
        Dual variable (potential).

    Raises
    ------
    ImportError
        If PyTorch is not installed.

    Notes
    -----
    This function uses PyTorch sparse tensors for memory efficiency.
    The API is different from the dense version to accommodate sparse
    data structures.

    For graphs with O(n) edges (e.g., planar graphs, grids), this can
    reduce memory usage from O(n²) to O(n).

    Examples
    --------
    >>> import numpy as np
    >>> # Create sparse cost matrix (COO format)
    >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> W_full = 1 / (A + 1e-9)
    >>> indices = np.array(np.nonzero(A > 0))
    >>> values = W_full[A > 0]
    >>> z = np.array([1.0, 0.0, -1.0])
    >>> # Run sparse Sinkhorn
    >>> f_idx, f_val, err, h = sinkhorn_w1_torch_sparse(
    ...     indices, values, (3, 3), z, epsilon=0.1, niter=100
    ... )
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for GPU acceleration. "
                          "Install it with: pip install torch")

    # Get device
    dev = get_device(device)

    n = W_shape[0]

    # Convert edge list to torch tensors (COO format)
    if isinstance(W_indices, np.ndarray):
        indices_torch = torch.from_numpy(W_indices).long().to(dev)
    else:
        indices_torch = W_indices.long().to(dev)

    if isinstance(W_values, np.ndarray):
        values_torch = torch.from_numpy(W_values).float().to(dev)
    else:
        values_torch = W_values.float().to(dev)

    if isinstance(z, np.ndarray):
        z_torch = torch.from_numpy(z).float().to(dev)
    else:
        z_torch = z.float().to(dev)

    # Sort edges by row index once (for fast segmented reductions)
    row_idx = indices_torch[0]
    col_idx = indices_torch[1]
    perm = torch.argsort(row_idx)
    invperm = torch.empty_like(perm)
    invperm[perm] = torch.arange(len(perm), device=dev)

    row_sorted = row_idx[perm]
    col_sorted = col_idx[perm]
    values_sorted = values_torch[perm]

    # CSR-style pointers for row segments
    row_counts = torch.bincount(row_sorted, minlength=n)
    row_ptr = torch.zeros(n + 1, dtype=torch.long, device=dev)
    row_ptr[1:] = torch.cumsum(row_counts, dim=0)

    # Initialize
    K_values_sorted = torch.exp(-values_sorted / epsilon)

    h = torch.zeros(n, dtype=torch.float32, device=dev)
    r = z_torch / 2
    err = []
    f_values_sorted = torch.exp(
        (-values_sorted + h[row_sorted] - h[col_sorted]) / epsilon
    )

    # Identify node types
    I0 = torch.where(z_torch == 0)[0]
    Ip = torch.where(z_torch > 0)[0]
    In = torch.where(z_torch < 0)[0]

    for it in range(niter):
        # Sparse matrix-vector products via scatter-add (no dense conversion)
        a = torch.zeros(n, dtype=torch.float32, device=dev)
        a.scatter_add_(0, row_sorted, K_values_sorted * torch.exp(-h[col_sorted] / epsilon))

        b = torch.zeros(n, dtype=torch.float32, device=dev)
        b.scatter_add_(0, row_sorted, K_values_sorted * torch.exp(+h[col_sorted] / epsilon))

        # Row-wise log-sum-exp from sparse edge values (no dense conversion)
        loga_vals = -values_sorted / epsilon - h[col_sorted] / epsilon
        logb_vals = -values_sorted / epsilon + h[col_sorted] / epsilon
        loga = _segment_logsumexp_from_sorted(loga_vals, row_ptr, n)
        logb = _segment_logsumexp_from_sorted(logb_vals, row_ptr, n)

        # Compute update vector m
        m = torch.zeros(n, dtype=torch.float32, device=dev)
        if len(I0) > 0:
            m[I0] = (loga[I0] - logb[I0]) / 2
        if len(Ip) > 0:
            m[Ip] = torch.log(torch.sqrt(r[Ip]**2 + a[Ip] * b[Ip]) + r[Ip]) - logb[Ip]
        if len(In) > 0:
            m[In] = -torch.log(torch.sqrt(r[In]**2 + a[In] * b[In]) - r[In]) + loga[In]

        # Update potential
        h = h / 2 - epsilon / 2 * m

        # Compute flow (sparse)
        f_values_sorted = torch.exp(
            (-values_sorted + h[row_sorted] - h[col_sorted]) / epsilon
        )

        # Compute error from sparse edge lists (no dense conversion)
        col_sums = torch.zeros(n, dtype=torch.float32, device=dev)
        col_sums.scatter_add_(0, col_sorted, f_values_sorted)
        row_sums = torch.zeros(n, dtype=torch.float32, device=dev)
        row_sums.scatter_add_(0, row_sorted, f_values_sorted)
        e = torch.norm((col_sums - row_sums) - z_torch, p=1)
        err.append(e.item())

    # Return f values in original edge ordering
    f_values = f_values_sorted[invperm]

    # Return results
    if return_numpy:
        return (indices_torch.cpu().numpy(),
                f_values.cpu().numpy(),
                err,
                h.cpu().numpy())
    else:
        return indices_torch, f_values, err, h


def check_gpu_availability():
    """
    Check GPU availability and print device information.

    Returns
    -------
    dict
        Dictionary with device information:
        - 'torch_available': bool
        - 'cuda_available': bool
        - 'mps_available': bool
        - 'device': torch.device
        - 'device_name': str
    """
    info = {
        'torch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'mps_available': False,
        'device': None,
        'device_name': 'N/A'
    }

    if not TORCH_AVAILABLE:
        return info

    info['cuda_available'] = torch.cuda.is_available()
    info['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    info['device'] = get_device()

    if info['cuda_available']:
        info['device_name'] = torch.cuda.get_device_name(0)
    elif info['mps_available']:
        info['device_name'] = 'Apple Metal Performance Shaders (MPS)'
    else:
        info['device_name'] = 'CPU'

    return info
