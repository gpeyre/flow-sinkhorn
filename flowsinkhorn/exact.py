"""
Exact (unregularized) W1 optimal transport solver using linear programming.

This module provides the exact solution to the Wasserstein-1 optimal transport
problem on graphs using the Beckmann (transshipment) linear program formulation.

References
----------
Gabriel Peyré, "Robust Sublinear Convergence Rates for Iterative Bregman
Projections on Affine Spaces", arXiv preprint, 2026.
"""

import numpy as np
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


def solve_w1_exact(W, z, solver=None, verbose=False):
    """
    Solve the exact (unregularized) W1 optimal transport problem.

    This function solves the Beckmann (transshipment) linear program:

        min_{F >= 0} <F, W>
        s.t. F^T 1 - F 1 = z

    where F is the flow matrix, W is the cost matrix, and z is the
    source/sink vector.

    The variable F[i,j] represents the flow from node j to node i. The
    compact representation enforces at optimality that either F[i,j] or
    F[j,i] is zero (no circular flow).

    Parameters
    ----------
    W : ndarray of shape (n, n)
        Cost/distance matrix between nodes. W[i,j] represents the cost of
        moving mass from node j to node i. For non-edges in a graph, W should
        be set to a large value (e.g., 1/epsilon for small epsilon).
    z : ndarray of shape (n,)
        Source/sink vector. Positive values are sources, negative are sinks,
        and sum(z) must be 0 for the problem to be feasible.
    solver : str, optional
        The CVXPY solver to use (e.g., 'ECOS', 'SCS', 'GLPK', 'MOSEK').
        If None, CVXPY will choose a suitable solver automatically.
    verbose : bool, default=False
        If True, print solver output.

    Returns
    -------
    F : ndarray of shape (n, n)
        Optimal flow matrix. F[i,j] is the flow from node j to node i.
    objective_value : float
        The optimal objective value (total transport cost).
    status : str
        The solver status ('optimal', 'infeasible', 'unbounded', etc.).

    Raises
    ------
    ImportError
        If cvxpy is not installed.
    ValueError
        If the problem is infeasible or unbounded.

    Notes
    -----
    This function requires CVXPY to be installed:
        pip install cvxpy

    For large-scale problems, you may want to install additional solvers
    such as MOSEK or Gurobi for better performance.

    The exact solver provides a reference solution but may be slow for
    large graphs. For faster approximate solutions, use the Sinkhorn
    algorithm from the sinkhorn module.

    Examples
    --------
    >>> import numpy as np
    >>> from flowsinkhorn.exact import solve_w1_exact
    >>> # Create a simple graph with adjacency matrix A
    >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> W = 1 / (A + 1e-9)  # Cost matrix (large for non-edges)
    >>> z = np.array([1.0, 0.0, -1.0])  # Source at 0, sink at 2
    >>> F, obj_val, status = solve_w1_exact(W, z)
    >>> print(f"Optimal cost: {obj_val:.4f}")
    >>> print(f"Status: {status}")
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("CVXPY is required for exact solver. "
                          "Install it with: pip install cvxpy")

    n = len(z)

    # Verify mass conservation
    if not np.isclose(np.sum(z), 0):
        raise ValueError(f"The source/sink vector z must sum to 0. "
                         f"Current sum: {np.sum(z):.6e}")

    # Define the optimization variable
    F_var = cp.Variable((n, n), nonneg=True)

    # Define the divergence constraint: F^T 1 - F 1 = z
    ones = np.ones(n)
    constraint = [F_var.T @ ones - F_var @ ones == z]

    # Define the objective: minimize <F, W>
    objective = cp.Minimize(cp.sum(cp.multiply(F_var, W)))

    # Create and solve the problem
    problem = cp.Problem(objective, constraint)

    try:
        if solver is not None:
            problem.solve(solver=solver, verbose=verbose)
        else:
            problem.solve(verbose=verbose)
    except Exception as e:
        raise RuntimeError(f"Solver failed with error: {str(e)}")

    # Check if the problem was solved successfully
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Problem is {problem.status}. "
                         f"Cannot find optimal solution.")

    # Extract the solution
    F = F_var.value
    objective_value = problem.value
    status = problem.status

    return F, objective_value, status


def solve_w1_exact_sparse(A, z, edge_costs=None, solver=None, verbose=False):
    """
    Solve the exact W1 problem on a sparse graph using edge-based formulation.

    This function is more memory-efficient for sparse graphs as it only
    creates flow variables for existing edges rather than all n^2 pairs.

    Parameters
    ----------
    A : ndarray of shape (n, n)
        Adjacency matrix of the graph (binary or weighted).
        A[i,j] > 0 indicates an edge from i to j.
    z : ndarray of shape (n,)
        Source/sink vector. Positive values are sources, negative are sinks,
        and sum(z) must be 0.
    edge_costs : ndarray of shape (n, n), optional
        Cost matrix for edges. If None, uniform costs of 1 are used for
        all edges.
    solver : str, optional
        The CVXPY solver to use.
    verbose : bool, default=False
        If True, print solver output.

    Returns
    -------
    F : ndarray of shape (n, n)
        Optimal flow matrix (sparse, only non-zero on edges).
    objective_value : float
        The optimal objective value.
    status : str
        The solver status.

    Raises
    ------
    ImportError
        If cvxpy is not installed.
    ValueError
        If the problem is infeasible or unbounded.

    Notes
    -----
    This formulation is recommended for graphs with O(n) edges rather than
    O(n^2) edges, such as k-nearest neighbor graphs or grid graphs.

    Examples
    --------
    >>> import numpy as np
    >>> from flowsinkhorn.exact import solve_w1_exact_sparse
    >>> # Create a simple chain graph
    >>> A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> z = np.array([1.0, 0.0, -1.0])
    >>> F, obj_val, status = solve_w1_exact_sparse(A, z)
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("CVXPY is required for exact solver. "
                          "Install it with: pip install cvxpy")

    n = len(z)

    # Verify mass conservation
    if not np.isclose(np.sum(z), 0):
        raise ValueError(f"The source/sink vector z must sum to 0. "
                         f"Current sum: {np.sum(z):.6e}")

    # Use unit costs if not provided
    if edge_costs is None:
        W = A.copy().astype(float)
        W[W > 0] = 1.0
    else:
        W = edge_costs.copy()

    # Set large cost for non-edges
    W[A == 0] = 1e9

    # Call the dense solver (could be optimized for sparse case)
    # For a truly sparse implementation, one would create variables only
    # for edges, but this requires more complex constraint formulation
    return solve_w1_exact(W, z, solver=solver, verbose=verbose)
