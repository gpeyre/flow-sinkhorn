"""Single-cell benchmark pipeline on Waddington-OT tutorial data.

This module provides:
1) Download/extract helpers for the WOT tutorial dataset.
2) Single-cell preprocessing in the spirit of WOT practice
   (log-normalization if needed, scaling, PCA).
3) Random per-time subsampling to keep experiments lightweight.
4) k-NN graph construction over all sampled cells.
5) Flow-Sinkhorn graph-OT benchmarking from mu_1 to mu_T and across
   adjacent snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse as sp
from scipy.sparse import csgraph
from scipy.special import logsumexp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from flowsinkhorn import sinkhorn_w1
from flowsinkhorn import sinkhorn_w1_sparse

try:
    import torch
except Exception:
    torch = None

try:
    from flowsinkhorn import sinkhorn_w1_torch_sparse
    _TORCH_SOLVER_AVAILABLE = True
except Exception:
    _TORCH_SOLVER_AVAILABLE = False

try:
    import anndata as ad
except Exception:
    ad = None

WOT_TUTORIAL_DATA_FILE_ID = "1E494DhIx5RLy0qv_6eWa9426Bfmq28po"
WOT_TUTORIAL_DATA_URL = (
    "https://drive.google.com/file/d/"
    f"{WOT_TUTORIAL_DATA_FILE_ID}/view?usp=drive_open"
)


@dataclass
class BenchmarkConfig:
    """Configuration for single-cell graph-OT benchmark."""

    data_root: Path = Path("data/wot")
    random_state: int = 0
    n_cells_per_time: int = 200
    pca_components: int = 30
    knn_k: int = 4
    epsilon: float = 0.05
    niter: int = 400
    use_torch: bool = True
    device: Optional[str] = None
    max_timepoints: Optional[int] = None


def download_wot_tutorial_data(
    data_root: Path | str,
    *,
    force: bool = False,
) -> Path:
    """Download WOT tutorial data.zip from Google Drive.

    Requires `gdown` for robust Google Drive downloads.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / "data.zip"

    if zip_path.exists() and not force:
        return zip_path

    try:
        import gdown
    except Exception as exc:
        raise ImportError(
            "gdown is required to download WOT tutorial data from Google Drive. "
            "Install with: pip install gdown"
        ) from exc

    gdown.download(id=WOT_TUTORIAL_DATA_FILE_ID, output=str(zip_path), quiet=False)
    return zip_path


def extract_zip(zip_path: Path | str, out_dir: Path | str) -> Path:
    """Extract ZIP archive and return extraction directory."""
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    return out_dir


def _read_cell_days(cell_days_path: Path) -> pd.Series:
    """Read WOT `cell_days.txt` robustly.

    Expected format: first column cell id, second column day value.
    """
    df = pd.read_csv(cell_days_path, sep=None, engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Could not parse cell days file: {cell_days_path}")

    days = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    valid = ~days.isna()
    out = pd.Series(days.loc[valid].to_numpy(), index=df.loc[valid, 0].astype(str).to_numpy())
    out.name = "day"
    return out


def _read_cell_filter(cell_filter_path: Optional[Path]) -> Optional[pd.Index]:
    if cell_filter_path is None or not cell_filter_path.exists():
        return None
    ids: List[str] = []
    with open(cell_filter_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ids.append(s.split()[0])
    if not ids:
        return None
    return pd.Index(np.array(ids, dtype=str))


def _locate_wot_files(data_root: Path) -> Dict[str, Path]:
    candidates = list(data_root.rglob("ExprMatrix.var.genes.h5ad"))
    if not candidates:
        candidates = list(data_root.rglob("ExprMatrix.h5ad"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find ExprMatrix.var.genes.h5ad (or ExprMatrix.h5ad)."
        )

    cell_days = list(data_root.rglob("cell_days.txt"))
    if not cell_days:
        raise FileNotFoundError("Could not find cell_days.txt in extracted data.")

    serum_filter = list(data_root.rglob("serum_cell_ids.txt"))
    out = {
        "matrix": candidates[0],
        "cell_days": cell_days[0],
    }
    if serum_filter:
        out["cell_filter"] = serum_filter[0]
    return out


def ensure_wot_data_available(data_root: Path | str) -> Dict[str, Path]:
    """Ensure WOT tutorial files exist locally, downloading if required."""
    data_root = Path(data_root)
    try:
        return _locate_wot_files(data_root)
    except FileNotFoundError:
        zip_path = download_wot_tutorial_data(data_root)
        extract_zip(zip_path, data_root)
        return _locate_wot_files(data_root)


def _sample_cells_by_time(
    obs_names: pd.Index,
    cell_days: pd.Series,
    *,
    n_cells_per_time: int,
    random_state: int,
    allowed_cells: Optional[pd.Index] = None,
    max_timepoints: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Subsample up to n cells per timepoint and return selected cell ids + days.

    Returns
    -------
    selected_cells
        Cell ids in sample order.
    days
        Day value per selected cell.
    unique_days
        Sorted unique days kept.
    """
    rng = np.random.default_rng(random_state)

    available = pd.Index(obs_names.astype(str))
    days = cell_days[cell_days.index.isin(available)]

    if allowed_cells is not None:
        days = days[days.index.isin(allowed_cells)]

    unique_days = np.array(sorted(days.unique()))
    if max_timepoints is not None:
        unique_days = unique_days[:max_timepoints]

    selected: List[str] = []
    selected_days: List[float] = []
    for d in unique_days:
        ids = days.index[days.values == d].to_numpy()
        if len(ids) == 0:
            continue
        if len(ids) > n_cells_per_time:
            ids = rng.choice(ids, size=n_cells_per_time, replace=False)
        selected.extend(ids.tolist())
        selected_days.extend([float(d)] * len(ids))

    if not selected:
        raise ValueError("No cells selected after filtering/subsampling.")

    return selected, np.asarray(selected_days), np.asarray(sorted(np.unique(selected_days)))


def _needs_log_normalization(X: sp.spmatrix | np.ndarray) -> bool:
    """Heuristic: detect count-like matrices."""
    if sp.issparse(X):
        data = X.data
        if data.size == 0:
            return False
        sample = data[: min(20000, data.size)]
    else:
        flat = np.ravel(X)
        if flat.size == 0:
            return False
        sample = flat[: min(20000, flat.size)]

    if np.nanmax(sample) <= 20:
        return False

    rounded = np.round(sample)
    is_integer_like = np.allclose(sample, rounded, atol=1e-6)
    return bool(is_integer_like and np.nanmin(sample) >= 0)


def _library_size_normalize_log1p(
    X: sp.spmatrix | np.ndarray,
    target_sum: float = 1e4,
) -> sp.spmatrix | np.ndarray:
    """Per-cell normalize total counts then log1p."""
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        lib = np.asarray(X.sum(axis=1)).ravel()
        scale = np.divide(target_sum, np.maximum(lib, 1e-12))
        X = sp.diags(scale) @ X
        X.data = np.log1p(X.data)
        return X

    X = np.asarray(X, dtype=np.float64)
    lib = X.sum(axis=1)
    scale = np.divide(target_sum, np.maximum(lib, 1e-12))
    X = X * scale[:, None]
    X = np.log1p(X)
    return X


def _compute_pca(
    X: sp.spmatrix | np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    n_features = X.shape[1]
    n_components = int(min(n_components, max(2, n_features - 1)))

    if sp.issparse(X):
        scaler = StandardScaler(with_mean=False)
        Xs = scaler.fit_transform(X)
        pca = TruncatedSVD(n_components=n_components, random_state=random_state)
        emb = pca.fit_transform(Xs)
    else:
        scaler = StandardScaler(with_mean=True)
        Xs = scaler.fit_transform(X)
        pca = PCA(n_components=n_components, random_state=random_state)
        emb = pca.fit_transform(Xs)

    return emb.astype(np.float64)


def _ensure_connected_graph_edges(
    embedding: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    val: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add minimal bridge edges so the undirected graph is connected."""
    if n <= 1:
        return row, col, val

    G = sp.csr_matrix((np.ones_like(val), (row, col)), shape=(n, n))
    n_comp, labels = csgraph.connected_components(G, directed=False, return_labels=True)
    if n_comp <= 1:
        return row, col, val

    comp_ids = np.arange(n_comp, dtype=np.int64)
    comp_sizes = np.bincount(labels, minlength=n_comp)
    root_comp = int(comp_ids[np.argmax(comp_sizes)])
    connected_nodes = np.where(labels == root_comp)[0]
    remaining = [int(c) for c in comp_ids.tolist() if int(c) != root_comp]

    add_rows: List[int] = []
    add_cols: List[int] = []
    add_vals: List[float] = []

    while remaining:
        nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nbrs.fit(embedding[connected_nodes])

        best = None
        best_dist = np.inf
        for comp in remaining:
            comp_nodes = np.where(labels == comp)[0]
            d, idx = nbrs.kneighbors(embedding[comp_nodes], return_distance=True)
            jloc = int(np.argmin(d[:, 0]))
            dist = float(d[jloc, 0])
            i_node = int(comp_nodes[jloc])
            j_node = int(connected_nodes[int(idx[jloc, 0])])
            if dist < best_dist:
                best_dist = dist
                best = (comp, i_node, j_node, dist)

        if best is None:
            break
        comp, i_node, j_node, dist = best
        w = float(max(dist, 1e-8))
        add_rows.extend([i_node, j_node])
        add_cols.extend([j_node, i_node])
        add_vals.extend([w, w])

        new_nodes = np.where(labels == comp)[0]
        connected_nodes = np.concatenate([connected_nodes, new_nodes])
        remaining = [c for c in remaining if c != comp]

    if add_rows:
        row = np.concatenate([row, np.asarray(add_rows, dtype=np.int64)])
        col = np.concatenate([col, np.asarray(add_cols, dtype=np.int64)])
        val = np.concatenate([val, np.asarray(add_vals, dtype=np.float64)])

    return row, col, val


def build_knn_graph(embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Construct a symmetric directed k-NN graph in COO format.

    Returns
    -------
    indices
        np.ndarray shape (2, nnz) of (row, col) directed edges.
    values
        np.ndarray shape (nnz,) edge costs (euclidean distance).
    shape
        Matrix shape (n, n).
    """
    n = embedding.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nbrs.fit(embedding)
    dists, neigh = nbrs.kneighbors(embedding, return_distance=True)

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []

    for i in range(n):
        js = neigh[i, 1:]
        ds = dists[i, 1:]
        rows.append(np.full_like(js, i, dtype=np.int64))
        cols.append(js.astype(np.int64))
        vals.append(np.maximum(ds.astype(np.float64), 1e-8))

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(vals)

    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    val_sym = np.concatenate([val, val])
    row_sym, col_sym, val_sym = _ensure_connected_graph_edges(
        embedding,
        row_sym.astype(np.int64),
        col_sym.astype(np.int64),
        val_sym.astype(np.float64),
        n,
    )

    W = sp.coo_matrix((val_sym, (row_sym, col_sym)), shape=(n, n)).tocsr()
    W = W.minimum(W.T)
    W = W.tocoo()

    indices = np.vstack([W.row.astype(np.int64), W.col.astype(np.int64)])
    values = W.data.astype(np.float64)
    return indices, values, (n, n)


def _build_snapshot_distributions(days: np.ndarray) -> Tuple[List[float], List[np.ndarray]]:
    unique_days = sorted(np.unique(days).tolist())
    n = len(days)

    mus: List[np.ndarray] = []
    for d in unique_days:
        mask = days == d
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mu = np.zeros(n, dtype=np.float64)
        mu[mask] = 1.0 / cnt
        mus.append(mu)

    return unique_days, mus


def _solve_pair_ot(
    mu_src: np.ndarray,
    mu_tgt: np.ndarray,
    *,
    W_indices: np.ndarray,
    W_values: np.ndarray,
    W_shape: Tuple[int, int],
    epsilon: float,
    niter: int,
    use_torch: bool,
    device: Optional[str],
) -> Dict[str, object]:
    z = mu_src - mu_tgt

    if use_torch and _TORCH_SOLVER_AVAILABLE:
        f_idx, f_val, err, h = sinkhorn_w1_torch_sparse(
            W_indices,
            W_values,
            W_shape,
            z,
            epsilon=epsilon,
            niter=niter,
            device=device,
            return_numpy=True,
        )
        cost = float(np.dot(f_val, W_values))
        return {
            "indices": f_idx,
            "flow_values": f_val,
            "err": err,
            "h": h,
            "cost": cost,
            "solver": "torch_sparse",
        }

    W_sparse = sp.coo_matrix((W_values, (W_indices[0], W_indices[1])), shape=W_shape)
    try:
        import sparse as pydata_sparse
    except Exception as exc:
        raise ImportError(
            "Need either PyTorch sparse solver or `sparse` package for CPU sparse solver."
        ) from exc

    W = pydata_sparse.COO.from_scipy_sparse(W_sparse)
    f, err, h = sinkhorn_w1_sparse(W, z, epsilon=epsilon, niter=niter)
    f_cost = float(np.dot(f.data, W.data))
    return {
        "indices": np.vstack([f.coords[0], f.coords[1]]),
        "flow_values": f.data,
        "err": err,
        "h": h,
        "cost": f_cost,
        "solver": "numpy_sparse",
    }


def _time_flux_profile(
    days: np.ndarray,
    edge_indices: np.ndarray,
    edge_flows: np.ndarray,
) -> pd.Series:
    n = len(days)
    flux = np.zeros(n, dtype=np.float64)
    rows = edge_indices[0]
    cols = edge_indices[1]
    np.add.at(flux, rows, edge_flows)
    np.add.at(flux, cols, edge_flows)
    flux *= 0.5

    df = pd.DataFrame({"day": days, "flux": flux})
    profile = df.groupby("day", sort=True)["flux"].sum()
    if profile.sum() > 0:
        profile = profile / profile.sum()
    return profile


def _empirical_time_profile(days: np.ndarray) -> pd.Series:
    prof = pd.Series(1.0, index=pd.Index(days, name="day")).groupby(level=0).sum()
    return prof / prof.sum()


def run_benchmark(config: BenchmarkConfig) -> Dict[str, object]:
    """Run full WOT single-cell graph-OT benchmark."""
    if ad is None:
        raise ImportError("anndata is required. Install with: pip install anndata")

    data_root = Path(config.data_root)
    files = _locate_wot_files(data_root)

    cell_days = _read_cell_days(files["cell_days"])
    cell_filter = _read_cell_filter(files.get("cell_filter"))

    adata_backed = ad.read_h5ad(files["matrix"], backed="r")
    selected_cells, sampled_days, unique_days = _sample_cells_by_time(
        pd.Index(adata_backed.obs_names.astype(str)),
        cell_days,
        n_cells_per_time=config.n_cells_per_time,
        random_state=config.random_state,
        allowed_cells=cell_filter,
        max_timepoints=config.max_timepoints,
    )

    adata = adata_backed[selected_cells, :].to_memory()
    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()

    X = adata.X
    if _needs_log_normalization(X):
        X = _library_size_normalize_log1p(X)

    emb = _compute_pca(
        X,
        n_components=config.pca_components,
        random_state=config.random_state,
    )

    W_indices, W_values, W_shape = build_knn_graph(emb, config.knn_k)
    day_values, mus = _build_snapshot_distributions(sampled_days)

    if len(mus) < 2:
        raise ValueError("Need at least two timepoints after subsampling.")

    direct = _solve_pair_ot(
        mus[0],
        mus[-1],
        W_indices=W_indices,
        W_values=W_values,
        W_shape=W_shape,
        epsilon=config.epsilon,
        niter=config.niter,
        use_torch=config.use_torch,
        device=config.device,
    )

    adjacent_costs: List[float] = []
    adjacent_details: List[Dict[str, object]] = []
    for t in range(len(mus) - 1):
        out = _solve_pair_ot(
            mus[t],
            mus[t + 1],
            W_indices=W_indices,
            W_values=W_values,
            W_shape=W_shape,
            epsilon=config.epsilon,
            niter=config.niter,
            use_torch=config.use_torch,
            device=config.device,
        )
        adjacent_costs.append(float(out["cost"]))
        adjacent_details.append(
            {
                "day_from": day_values[t],
                "day_to": day_values[t + 1],
                "cost": float(out["cost"]),
                "final_error": float(out["err"][-1]),
            }
        )

    cumulative = float(np.sum(adjacent_costs))
    ratio = float(direct["cost"] / cumulative) if cumulative > 0 else np.nan

    direct_profile = _time_flux_profile(
        sampled_days,
        direct["indices"],
        direct["flow_values"],
    )
    empirical_profile = _empirical_time_profile(sampled_days)
    profile_df = pd.concat(
        [empirical_profile.rename("empirical"), direct_profile.rename("ot_flux")], axis=1
    ).fillna(0.0)
    profile_l1 = float(np.abs(profile_df["empirical"] - profile_df["ot_flux"]).sum())

    return {
        "config": config,
        "n_cells": int(emb.shape[0]),
        "n_timepoints": int(len(day_values)),
        "days": day_values,
        "selected_cells": selected_cells,
        "sampled_days": sampled_days,
        "embedding": emb,
        "graph": {
            "indices": W_indices,
            "values": W_values,
            "shape": W_shape,
            "nnz": int(W_values.size),
        },
        "direct": {
            "cost": float(direct["cost"]),
            "final_error": float(direct["err"][-1]),
            "solver": direct["solver"],
            "indices": direct["indices"],
            "flow_values": direct["flow_values"],
            "err": direct["err"],
        },
        "adjacent": adjacent_details,
        "cumulative_adjacent_cost": cumulative,
        "geodesic_ratio": ratio,
        "time_profile": profile_df,
        "time_profile_l1": profile_l1,
    }


def summarize_benchmark(result: Dict[str, object]) -> pd.DataFrame:
    """Return a compact tabular summary of benchmark outputs."""
    summary = pd.DataFrame(
        {
            "metric": [
                "n_cells",
                "n_timepoints",
                "direct_cost_mu1_to_muT",
                "sum_adjacent_costs",
                "geodesic_ratio_direct_over_path",
                "direct_final_constraint_error",
                "time_profile_l1",
            ],
            "value": [
                result["n_cells"],
                result["n_timepoints"],
                result["direct"]["cost"],
                result["cumulative_adjacent_cost"],
                result["geodesic_ratio"],
                result["direct"]["final_error"],
                result["time_profile_l1"],
            ],
        }
    )
    return summary


def prepare_wot_data(config: BenchmarkConfig) -> Dict[str, object]:
    """Prepare sampled WOT data, PCA embedding, graph and snapshot measures."""
    if ad is None:
        raise ImportError("anndata is required. Install with: pip install anndata")

    data_root = Path(config.data_root)
    files = ensure_wot_data_available(data_root)
    cell_days = _read_cell_days(files["cell_days"])
    cell_filter = _read_cell_filter(files.get("cell_filter"))

    adata_backed = ad.read_h5ad(files["matrix"], backed="r")
    selected_cells, sampled_days, unique_days = _sample_cells_by_time(
        pd.Index(adata_backed.obs_names.astype(str)),
        cell_days,
        n_cells_per_time=config.n_cells_per_time,
        random_state=config.random_state,
        allowed_cells=cell_filter,
        max_timepoints=config.max_timepoints,
    )
    adata = adata_backed[selected_cells, :].to_memory()
    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()

    X = adata.X
    if _needs_log_normalization(X):
        X = _library_size_normalize_log1p(X)
    emb = _compute_pca(X, n_components=config.pca_components, random_state=config.random_state)

    W_indices, W_values, W_shape = build_knn_graph(emb, config.knn_k)
    day_values, mus = _build_snapshot_distributions(sampled_days)
    day_to_indices = {
        float(d): np.where(sampled_days == float(d))[0]
        for d in day_values
    }
    return {
        "config": config,
        "selected_cells": selected_cells,
        "sampled_days": sampled_days,
        "days": day_values,
        "day_to_indices": day_to_indices,
        "mus": mus,
        "embedding": emb,
        "graph_indices": W_indices,
        "graph_values": W_values,
        "graph_shape": W_shape,
        "n_cells": int(emb.shape[0]),
    }


def floyd_warshall_metric(
    graph_indices: np.ndarray,
    graph_values: np.ndarray,
    graph_shape: Tuple[int, int],
    *,
    return_predecessors: bool = False,
) -> Dict[str, object]:
    """Compute all-pairs shortest path metric with timed Floyd-Warshall."""
    G = sp.csr_matrix(
        (graph_values, (graph_indices[0], graph_indices[1])),
        shape=graph_shape,
        dtype=np.float64,
    )
    t0 = time.perf_counter()
    if return_predecessors:
        D, pred = csgraph.floyd_warshall(
            G, directed=False, return_predecessors=True
        )
    else:
        D = csgraph.floyd_warshall(G, directed=False, return_predecessors=False)
        pred = None
    dt = time.perf_counter() - t0
    return {"distances": D, "predecessors": pred, "time_sec": dt}


def solve_graph_w1_lp(
    graph_indices: np.ndarray,
    graph_values: np.ndarray,
    graph_shape: Tuple[int, int],
    z: np.ndarray,
) -> Dict[str, object]:
    """Exact graph W1 via sparse LP (HiGHS) on directed graph edges."""
    n = graph_shape[0]
    m = graph_values.size
    if z.shape[0] != n:
        raise ValueError(f"z has length {z.shape[0]}, expected {n}")

    rows = graph_indices[0].astype(np.int64)
    cols = graph_indices[1].astype(np.int64)
    edge_ids = np.arange(m, dtype=np.int64)

    # Match Sinkhorn convention used in this project:
    #   sum_in - sum_out = z
    data = np.concatenate([-np.ones(m), np.ones(m)])
    r_idx = np.concatenate([rows, cols])
    c_idx = np.concatenate([edge_ids, edge_ids])
    A_eq = sp.csr_matrix((data, (r_idx, c_idx)), shape=(n, m))

    t0 = time.perf_counter()
    res = linprog(
        c=graph_values.astype(np.float64),
        A_eq=A_eq,
        b_eq=z.astype(np.float64),
        bounds=(0, None),
        method="highs",
    )
    dt = time.perf_counter() - t0

    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    dual = None
    try:
        dual = solve_graph_w1_dual_lp(graph_indices, graph_values, graph_shape, z)["dual_potential"]
    except Exception:
        if hasattr(res, "eqlin") and hasattr(res.eqlin, "marginals"):
            dual = np.asarray(res.eqlin.marginals, dtype=np.float64)

    return {
        "flow_values": np.asarray(res.x, dtype=np.float64),
        "objective": float(res.fun),
        "dual_potential": dual,
        "status": res.status,
        "message": res.message,
        "nit": getattr(res, "nit", None),
        "time_sec": dt,
    }


def solve_graph_w1_dual_lp(
    graph_indices: np.ndarray,
    graph_values: np.ndarray,
    graph_shape: Tuple[int, int],
    z: np.ndarray,
) -> Dict[str, object]:
    """Exact graph W1 dual potential via LP.

    Solve:
        max_h <z, h>
        s.t. h_j - h_i <= w_{ij} for each directed edge (i, j)
    with gauge fixing h_0 = 0.
    """
    n = graph_shape[0]
    m = graph_values.size
    if z.shape[0] != n:
        raise ValueError(f"z has length {z.shape[0]}, expected {n}")

    rows = graph_indices[0].astype(np.int64)
    cols = graph_indices[1].astype(np.int64)
    edge_ids = np.arange(m, dtype=np.int64)

    data = np.concatenate([-np.ones(m), np.ones(m)])
    r_idx = np.concatenate([edge_ids, edge_ids])
    c_idx = np.concatenate([rows, cols])
    A_ub = sp.csr_matrix((data, (r_idx, c_idx)), shape=(m, n))
    b_ub = graph_values.astype(np.float64)

    A_eq = sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n))
    b_eq = np.array([0.0], dtype=np.float64)

    t0 = time.perf_counter()
    res = linprog(
        c=(-z).astype(np.float64),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * n,
        method="highs",
    )
    dt = time.perf_counter() - t0

    if not res.success:
        raise RuntimeError(f"Dual LP failed: {res.message}")

    h = np.asarray(res.x, dtype=np.float64)
    return {
        "dual_potential": h,
        "objective": float(np.dot(z, h)),
        "status": res.status,
        "message": res.message,
        "time_sec": dt,
    }


def solve_line_w1_dual_closed_form(
    z: np.ndarray,
    *,
    gauge: str = "zero_at_first",
) -> Dict[str, object]:
    """Closed-form dual optimal potential for 1D path metric W1.

    For c_i = sum_{j<=i} z_j (i=0..n-2), choose slope d_i = -sign(c_i), then
    h_{i+1} - h_i = d_i is dual-optimal under |h_{i+1}-h_i|<=1.
    """
    zz = np.asarray(z, dtype=np.float64).ravel()
    n = int(zz.size)
    if n == 0:
        return {"dual_potential": np.array([], dtype=np.float64), "objective": 0.0}

    c = np.cumsum(zz)
    slopes = -np.sign(c[:-1]) if n > 1 else np.array([], dtype=np.float64)
    h = np.zeros(n, dtype=np.float64)
    if n > 1:
        h[1:] = np.cumsum(slopes)

    if gauge == "centered":
        h = h - float(np.mean(h))
    elif gauge != "zero_at_first":
        raise ValueError(f"Unknown gauge: {gauge}")

    w1 = float(np.sum(np.abs(c[:-1]))) if n > 1 else 0.0
    obj = float(np.dot(zz, h))
    # Numerical guard: objective should match closed-form W1 exactly up to fp noise.
    if abs(obj - w1) > 1e-10 * max(1.0, w1):
        raise RuntimeError(
            f"Closed-form line dual construction failed: obj={obj:.12g}, w1={w1:.12g}"
        )
    return {"dual_potential": h, "objective": obj}


def solve_flow_sinkhorn_sparse(
    graph_indices: np.ndarray,
    graph_values: np.ndarray,
    graph_shape: Tuple[int, int],
    z: np.ndarray,
    *,
    epsilon: float,
    niter: int,
    use_torch: bool = True,
    device: Optional[str] = None,
    h_clip: Optional[float] = 5000.0,
    relaxation: float = 0.8,
) -> Dict[str, object]:
    """Run Flow-Sinkhorn on sparse graph and time it."""
    t0 = time.perf_counter()
    if use_torch:
        out = _flow_sinkhorn_sparse_iterations_torch(
            graph_indices,
            graph_values,
            graph_shape,
            z,
            epsilon=epsilon,
            niter=niter,
            record_every=max(1, niter),
            lp_dual=None,
            h_clip=h_clip,
            relaxation=relaxation,
            device=device,
            require_gpu=False,
        )
        solver = "torch_sparse"
    else:
        out = _flow_sinkhorn_sparse_iterations(
            graph_indices,
            graph_values,
            graph_shape,
            z,
            epsilon=epsilon,
            niter=niter,
            record_every=max(1, niter),
            lp_dual=None,
            h_clip=h_clip,
            relaxation=relaxation,
        )
        solver = "numpy_stabilized"
    idx_out = out["edge_indices"]
    flow_values = out["flow_values"]
    h_out = out["h"]
    err_curve = out["trajectory"]["constraint_l1"].to_list() if len(out["trajectory"]) > 0 else []
    dt = time.perf_counter() - t0
    completed = int(out.get("completed_iters", 0))
    crashed_iter = int(out.get("crashed_iter", 0))
    if completed <= 0:
        residual = np.nan
        objective = np.nan
    else:
        residual = _divergence_residual_l1(
            idx_out, flow_values, z, graph_shape[0]
        )
        objective = float(np.dot(flow_values, graph_values))
    return {
        "edge_indices": idx_out,
        "flow_values": flow_values,
        "h": h_out,
        "objective": objective,
        "constraint_l1": residual,
        "time_sec": dt,
        "solver": solver,
        "err_curve": err_curve,
        "completed_iters": completed,
        "crashed_iter": crashed_iter,
        "ops_total_theory": float(out.get("ops_total_theory", np.nan)),
        "ops_per_iter_theory": float(out.get("ops_per_iter_theory", np.nan)),
    }


def solve_vanilla_sinkhorn_dense(
    distance_matrix: np.ndarray,
    z: np.ndarray,
    *,
    epsilon: float,
    niter: int,
    h_clip: Optional[float] = 5000.0,
    relaxation: float = 0.5,
) -> Dict[str, object]:
    """Run dense (vanilla) Sinkhorn on full metric and time it."""
    t0 = time.perf_counter()
    out = _vanilla_sinkhorn_dense_iterations(
        distance_matrix,
        z,
        epsilon=epsilon,
        niter=niter,
        record_every=max(1, niter),
        lp_dual=None,
        h_clip=h_clip,
        relaxation=relaxation,
    )
    dt = time.perf_counter() - t0
    f = out["flow"]
    h = out["h"]
    completed = int(out.get("completed_iters", 0))
    crashed_iter = int(out.get("crashed_iter", 0))
    if completed <= 0:
        obj = np.nan
        residual = np.nan
    else:
        obj = float(np.sum(f * np.asarray(distance_matrix, dtype=np.float64)))
        residual = float(np.linalg.norm(f.sum(axis=0) - f.sum(axis=1) - z, ord=1))
    return {
        "flow": f,
        "h": np.asarray(h, dtype=np.float64),
        "objective": obj,
        "constraint_l1": residual,
        "time_sec": dt,
        "err_curve": out["trajectory"]["constraint_l1"].to_list() if len(out["trajectory"]) > 0 else [],
        "completed_iters": completed,
        "crashed_iter": crashed_iter,
        "ops_total_theory": float(out.get("ops_total_theory", np.nan)),
        "ops_per_iter_theory": float(out.get("ops_per_iter_theory", np.nan)),
    }


def _center_potential(p: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if p is None:
        return None
    p = np.asarray(p, dtype=np.float64)
    finite = np.isfinite(p)
    if not np.any(finite):
        return None
    return p - float(np.nanmean(p[finite]))


def dual_linf_error(
    h_est: Optional[np.ndarray],
    h_ref: Optional[np.ndarray],
    *,
    epsilon: Optional[float] = None,
    align_sign: bool = True,
) -> float:
    """L-infinity error between centered dual potentials."""
    if h_est is None or h_ref is None:
        return np.nan
    h0 = np.asarray(h_est, dtype=np.float64)
    if epsilon is not None and epsilon > 0:
        h0 = h0 / float(epsilon)
    a = _center_potential(h0)
    b = _center_potential(h_ref)
    if a is None or b is None:
        return np.nan
    if a.shape != b.shape:
        return np.nan
    e1 = float(np.max(np.abs(a - b)))
    if not align_sign:
        return e1
    e2 = float(np.max(np.abs(a + b)))
    return min(e1, e2)


def dual_l2_error(
    h_est: Optional[np.ndarray],
    h_ref: Optional[np.ndarray],
    *,
    epsilon: Optional[float] = None,
    align_sign: bool = True,
) -> float:
    """L2 error between centered dual potentials.

    Potentials are centered (mean removed) before comparison to account
    for the additive-constant gauge freedom.
    """
    if h_est is None or h_ref is None:
        return np.nan
    h0 = np.asarray(h_est, dtype=np.float64)
    if epsilon is not None and epsilon > 0:
        h0 = h0 / float(epsilon)
    a = _center_potential(h0)
    b = _center_potential(h_ref)
    if a is None or b is None:
        return np.nan
    if a.shape != b.shape:
        return np.nan
    e1 = float(np.linalg.norm(a - b, ord=2))
    if not align_sign:
        return e1
    e2 = float(np.linalg.norm(a + b, ord=2))
    return min(e1, e2)


def _w1_lipschitz_debias_potential(h: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    """Return a 1-Lipschitz c-transform debiasing of a W1 potential."""
    h0 = np.asarray(h, dtype=np.float64)
    D = np.asarray(distance_matrix, dtype=np.float64)
    upper = np.min(h0[None, :] + D, axis=1)
    lower = np.max(h0[None, :] - D, axis=1)
    return 0.5 * (upper + lower)


def _build_row_segments(
    edge_indices: np.ndarray,
    edge_values: np.ndarray,
    n: int,
) -> Dict[str, np.ndarray]:
    """Sort edges by row and build CSR-style row pointers."""
    rows = edge_indices[0].astype(np.int64)
    cols = edge_indices[1].astype(np.int64)
    vals = edge_values.astype(np.float64)
    perm = np.argsort(rows)
    rows_s = rows[perm]
    cols_s = cols[perm]
    vals_s = vals[perm]
    counts = np.bincount(rows_s, minlength=n)
    row_ptr = np.zeros(n + 1, dtype=np.int64)
    row_ptr[1:] = np.cumsum(counts)
    return {"rows": rows_s, "cols": cols_s, "vals": vals_s, "row_ptr": row_ptr}


def _segment_logsumexp_np(values_sorted: np.ndarray, row_ptr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -np.inf, dtype=np.float64)
    for i in range(n):
        a = row_ptr[i]
        b = row_ptr[i + 1]
        if b > a:
            out[i] = logsumexp(values_sorted[a:b])
    return out


def _theoretical_ops_per_iter_flow_sparse(n: int, m: int) -> float:
    """Elementary-op proxy count per Flow-Sinkhorn sparse iteration."""
    return float(18.0 * m + 20.0 * n)


def _theoretical_ops_per_iter_vanilla_dense(n: int) -> float:
    """Elementary-op proxy count per vanilla dense Sinkhorn iteration."""
    return float(22.0 * n * n + 20.0 * n)


def _safe_exp(x: np.ndarray, *, lo: float = -700.0, hi: float = 700.0) -> np.ndarray:
    return np.exp(np.clip(x, lo, hi))


def _center_in_place(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    if np.any(finite):
        x = x.copy()
        x[finite] -= float(np.mean(x[finite]))
    return x


def _asinh_exp_from_log(log_x: np.ndarray) -> np.ndarray:
    """Compute asinh(exp(log_x)) stably."""
    t = np.asarray(log_x, dtype=np.float64)
    out = np.empty_like(t)
    lo = t < -20.0
    hi = t > 20.0
    mid = ~(lo | hi)
    if np.any(lo):
        out[lo] = np.exp(t[lo])
    if np.any(hi):
        out[hi] = t[hi] + np.log(2.0)
    if np.any(mid):
        out[mid] = np.arcsinh(np.exp(t[mid]))
    return out


def _require_torch_device(device: Optional[str], require_gpu: bool) -> "torch.device":
    if torch is None:
        raise ImportError("PyTorch is required for GPU-only benchmark execution.")
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    if require_gpu and dev.type != "cuda":
        raise RuntimeError(f"GPU required but selected device is '{dev}'.")
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required but CUDA is not available in PyTorch.")
    return dev


def _flow_sinkhorn_sparse_iterations_torch(
    graph_indices: np.ndarray,
    graph_values: np.ndarray,
    graph_shape: Tuple[int, int],
    z: np.ndarray,
    *,
    epsilon: float,
    niter: int,
    record_every: int,
    lp_dual: Optional[np.ndarray],
    distance_matrix: Optional[np.ndarray] = None,
    h_clip: Optional[float] = 5000.0,
    relaxation: float = 0.8,
    device: Optional[str] = None,
    require_gpu: bool = False,
) -> Dict[str, object]:
    dev = _require_torch_device(device=device, require_gpu=require_gpu)
    n = int(graph_shape[0])

    rows = torch.as_tensor(graph_indices[0], dtype=torch.long, device=dev)
    cols = torch.as_tensor(graph_indices[1], dtype=torch.long, device=dev)
    w = torch.as_tensor(graph_values, dtype=torch.float64, device=dev)
    zt = torch.as_tensor(z, dtype=torch.float64, device=dev)
    eps = float(max(epsilon, 1e-12))
    relax = float(np.clip(relaxation, 1e-6, 1.0))

    perm = torch.argsort(rows)
    invperm = torch.empty_like(perm)
    invperm[perm] = torch.arange(len(perm), device=dev)
    row_sorted = rows[perm]
    col_sorted = cols[perm]
    w_sorted = w[perm]

    row_counts = torch.bincount(row_sorted, minlength=n)
    row_ptr = torch.zeros(n + 1, dtype=torch.long, device=dev)
    row_ptr[1:] = torch.cumsum(row_counts, dim=0)
    k_sorted = torch.exp(-w_sorted / eps)

    h = torch.zeros(n, dtype=torch.float64, device=dev)
    r = zt / 2.0
    I0 = torch.where(zt == 0)[0]
    Ip = torch.where(zt > 0)[0]
    In = torch.where(zt < 0)[0]

    last_h = h.clone()
    last_f_sorted = torch.zeros_like(w_sorted)
    last_iter = 0
    crashed_iter = 0

    D = None if distance_matrix is None else np.asarray(distance_matrix, dtype=np.float64)
    edge_idx = np.vstack([graph_indices[0], graph_indices[1]])
    ops_per_iter = _theoretical_ops_per_iter_flow_sparse(n, int(len(graph_values)))

    rows_out: List[Dict[str, float]] = []
    rows_out.append(
        {
            "iter": 0.0,
            "runtime_sec": 0.0,
            "objective": np.nan,
            "constraint_l1": np.nan,
            "dual_l2_vs_lp": dual_l2_error(np.zeros(n, dtype=np.float64), lp_dual),
            "dual_l2_debiased_vs_lp": (
                dual_l2_error(_w1_lipschitz_debias_potential(np.zeros(n, dtype=np.float64), D), lp_dual)
                if D is not None
                else np.nan
            ),
            "dual_linf_vs_lp": dual_linf_error(np.zeros(n, dtype=np.float64), lp_dual),
            "stable": 1.0,
            "ops_per_iter_theory": float(ops_per_iter),
            "ops_cum_theory": 0.0,
        }
    )

    t_start = time.perf_counter()
    for it in range(1, int(niter) + 1):
        try:
            a = torch.zeros(n, dtype=torch.float64, device=dev)
            a.scatter_add_(0, row_sorted, k_sorted * torch.exp(-h[col_sorted] / eps))
            b = torch.zeros(n, dtype=torch.float64, device=dev)
            b.scatter_add_(0, row_sorted, k_sorted * torch.exp(+h[col_sorted] / eps))

            loga_vals = -w_sorted / eps - h[col_sorted] / eps
            logb_vals = -w_sorted / eps + h[col_sorted] / eps
            # Fully vectorized segmented logsumexp via scatter-reduce (no Python row loop).
            neg_inf = torch.tensor(-torch.inf, dtype=torch.float64, device=dev)
            max_loga = torch.full((n,), neg_inf, dtype=torch.float64, device=dev)
            max_logb = torch.full((n,), neg_inf, dtype=torch.float64, device=dev)
            max_loga.scatter_reduce_(0, row_sorted, loga_vals, reduce="amax", include_self=True)
            max_logb.scatter_reduce_(0, row_sorted, logb_vals, reduce="amax", include_self=True)

            sumexp_loga = torch.zeros(n, dtype=torch.float64, device=dev)
            sumexp_logb = torch.zeros(n, dtype=torch.float64, device=dev)
            sumexp_loga.scatter_add_(0, row_sorted, torch.exp(loga_vals - max_loga[row_sorted]))
            sumexp_logb.scatter_add_(0, row_sorted, torch.exp(logb_vals - max_logb[row_sorted]))
            loga = max_loga + torch.log(sumexp_loga.clamp_min(1e-300))
            logb = max_logb + torch.log(sumexp_logb.clamp_min(1e-300))

            m = torch.zeros(n, dtype=torch.float64, device=dev)
            if len(I0) > 0:
                m[I0] = 0.5 * (loga[I0] - logb[I0])
            if len(Ip) > 0:
                m[Ip] = torch.log(torch.sqrt(r[Ip] ** 2 + a[Ip] * b[Ip]) + r[Ip]) - logb[Ip]
            if len(In) > 0:
                m[In] = -torch.log(torch.sqrt(r[In] ** 2 + a[In] * b[In]) - r[In]) + loga[In]

            h = (1.0 - relax) * h - relax * eps * m
            h = h - torch.mean(h)
            if h_clip is not None:
                hc = float(abs(h_clip))
                h = torch.clamp(h, -hc, hc)
            f_sorted = torch.exp((-w_sorted + h[row_sorted] - h[col_sorted]) / eps)
        except Exception:
            crashed_iter = it
            break

        if not bool(torch.isfinite(f_sorted).all().item()) or not bool(torch.isfinite(h).all().item()):
            crashed_iter = it
            break

        last_h = h.clone()
        last_f_sorted = f_sorted.clone()
        last_iter = it

        if (it % max(1, int(record_every)) == 0) or (it == int(niter)):
            h_np = h.detach().cpu().numpy()
            f_vals = f_sorted[invperm].detach().cpu().numpy()
            elapsed = time.perf_counter() - t_start
            rows_out.append(
                {
                    "iter": float(it),
                    "runtime_sec": float(elapsed),
                    "objective": float(np.dot(f_vals, graph_values)),
                    "constraint_l1": _divergence_residual_l1(edge_idx, f_vals, np.asarray(z, dtype=np.float64), n),
                    "dual_l2_vs_lp": dual_l2_error(h_np, lp_dual),
                    "dual_l2_debiased_vs_lp": (
                        dual_l2_error(_w1_lipschitz_debias_potential(h_np, D), lp_dual)
                        if D is not None
                        else np.nan
                    ),
                    "dual_linf_vs_lp": dual_linf_error(h_np, lp_dual),
                    "stable": 1.0,
                    "ops_per_iter_theory": float(ops_per_iter),
                    "ops_cum_theory": float(it) * float(ops_per_iter),
                }
            )

    if crashed_iter > 0:
        rows_out.append(
            {
                "iter": float(crashed_iter),
                "runtime_sec": float(time.perf_counter() - t_start),
                "objective": np.nan,
                "constraint_l1": np.nan,
                "dual_l2_vs_lp": np.nan,
                "dual_l2_debiased_vs_lp": np.nan,
                "dual_linf_vs_lp": np.nan,
                "stable": 0.0,
                "ops_per_iter_theory": float(ops_per_iter),
                "ops_cum_theory": float(crashed_iter) * float(ops_per_iter),
            }
        )

    return {
        "trajectory": pd.DataFrame(rows_out),
        "h": last_h.detach().cpu().numpy(),
        "flow_values": last_f_sorted[invperm].detach().cpu().numpy(),
        "edge_indices": edge_idx,
        "completed_iters": int(last_iter),
        "crashed_iter": int(crashed_iter),
        "ops_per_iter_theory": float(ops_per_iter),
        "ops_total_theory": float(last_iter) * float(ops_per_iter),
    }


def _vanilla_sinkhorn_dense_iterations_torch(
    distance_matrix: np.ndarray,
    z: np.ndarray,
    *,
    epsilon: float,
    niter: int,
    record_every: int,
    lp_dual: Optional[np.ndarray],
    h_clip: Optional[float] = 5000.0,
    relaxation: float = 0.5,
    device: Optional[str] = None,
    require_gpu: bool = False,
) -> Dict[str, object]:
    dev = _require_torch_device(device=device, require_gpu=require_gpu)
    W = torch.as_tensor(distance_matrix, dtype=torch.float64, device=dev)
    zt = torch.as_tensor(z, dtype=torch.float64, device=dev)
    n = int(zt.numel())
    h = torch.zeros(n, dtype=torch.float64, device=dev)
    r = zt / 2.0
    I0 = torch.where(zt == 0)[0]
    Ip = torch.where(zt > 0)[0]
    In = torch.where(zt < 0)[0]

    eps = float(max(epsilon, 1e-12))
    relax = float(np.clip(relaxation, 1e-6, 1.0))
    last_h = h.clone()
    last_f = torch.zeros((n, n), dtype=torch.float64, device=dev)
    last_iter = 0
    crashed_iter = 0
    ops_per_iter = _theoretical_ops_per_iter_vanilla_dense(n)

    rows_out: List[Dict[str, float]] = []
    rows_out.append(
        {
            "iter": 0.0,
            "runtime_sec": 0.0,
            "objective": np.nan,
            "constraint_l1": np.nan,
            "dual_l2_vs_lp": dual_l2_error(np.zeros(n, dtype=np.float64), lp_dual),
            "dual_linf_vs_lp": dual_linf_error(np.zeros(n, dtype=np.float64), lp_dual),
            "stable": 1.0,
            "ops_per_iter_theory": float(ops_per_iter),
            "ops_cum_theory": 0.0,
        }
    )

    t_start = time.perf_counter()
    for it in range(1, int(niter) + 1):
        try:
            loga = torch.logsumexp(-W / eps + (-h / eps)[None, :], dim=1)
            logb = torch.logsumexp(-W / eps + (+h / eps)[None, :], dim=1)
            m = torch.zeros(n, dtype=torch.float64, device=dev)
            if len(I0) > 0:
                m[I0] = 0.5 * (loga[I0] - logb[I0])
            if len(Ip) > 0:
                m[Ip] = torch.log(torch.sqrt(r[Ip] ** 2 + torch.exp(loga[Ip]) * torch.exp(logb[Ip])) + r[Ip]) - logb[Ip]
            if len(In) > 0:
                m[In] = -torch.log(torch.sqrt(r[In] ** 2 + torch.exp(loga[In]) * torch.exp(logb[In])) - r[In]) + loga[In]
            h = (1.0 - relax) * h - relax * eps * m
            h = h - torch.mean(h)
            if h_clip is not None:
                hc = float(abs(h_clip))
                h = torch.clamp(h, -hc, hc)
            f = torch.exp((-W + h[:, None] - h[None, :]) / eps)
        except Exception:
            crashed_iter = it
            break

        if not bool(torch.isfinite(f).all().item()) or not bool(torch.isfinite(h).all().item()):
            crashed_iter = it
            break

        last_h = h.clone()
        last_f = f.clone()
        last_iter = it

        if (it % max(1, int(record_every)) == 0) or (it == int(niter)):
            h_np = h.detach().cpu().numpy()
            f_np = f.detach().cpu().numpy()
            z_np = zt.detach().cpu().numpy()
            elapsed = time.perf_counter() - t_start
            rows_out.append(
                {
                    "iter": float(it),
                    "runtime_sec": float(elapsed),
                    "objective": float(np.sum(f_np * np.asarray(distance_matrix, dtype=np.float64))),
                    "constraint_l1": float(np.linalg.norm((np.sum(f_np, axis=0) - np.sum(f_np, axis=1)) - z_np, ord=1)),
                    "dual_l2_vs_lp": dual_l2_error(h_np, lp_dual),
                    "dual_linf_vs_lp": dual_linf_error(h_np, lp_dual),
                    "stable": 1.0,
                    "ops_per_iter_theory": float(ops_per_iter),
                    "ops_cum_theory": float(it) * float(ops_per_iter),
                }
            )

    if crashed_iter > 0:
        rows_out.append(
            {
                "iter": float(crashed_iter),
                "runtime_sec": float(time.perf_counter() - t_start),
                "objective": np.nan,
                "constraint_l1": np.nan,
                "dual_l2_vs_lp": np.nan,
                "dual_linf_vs_lp": np.nan,
                "stable": 0.0,
                "ops_per_iter_theory": float(ops_per_iter),
                "ops_cum_theory": float(crashed_iter) * float(ops_per_iter),
            }
        )

    return {
        "trajectory": pd.DataFrame(rows_out),
        "h": last_h.detach().cpu().numpy(),
        "flow": last_f.detach().cpu().numpy(),
        "completed_iters": int(last_iter),
        "crashed_iter": int(crashed_iter),
        "ops_per_iter_theory": float(ops_per_iter),
        "ops_total_theory": float(last_iter) * float(ops_per_iter),
    }


def _flow_sinkhorn_sparse_iterations(
    graph_indices: np.ndarray,
    graph_values: np.ndarray,
    graph_shape: Tuple[int, int],
    z: np.ndarray,
    *,
    epsilon: float,
    niter: int,
    record_every: int,
    lp_dual: Optional[np.ndarray],
    distance_matrix: Optional[np.ndarray] = None,
    h_clip: Optional[float] = 5000.0,
    relaxation: float = 0.8,
) -> Dict[str, object]:
    n = graph_shape[0]
    seg = _build_row_segments(graph_indices, graph_values, n)
    rows = seg["rows"]
    cols = seg["cols"]
    w = seg["vals"]
    row_ptr = seg["row_ptr"]
    D = None if distance_matrix is None else np.asarray(distance_matrix, dtype=np.float64)

    zf = z.astype(np.float64)
    r = zf / 2.0
    I0 = np.where(zf == 0)[0]
    Ip = np.where(zf > 0)[0]
    In = np.where(zf < 0)[0]

    h = np.zeros(n, dtype=np.float64)
    last_h = h.copy()
    last_f = np.zeros_like(w)
    last_iter = 0
    crashed_iter = 0

    t_start = time.perf_counter()
    rows_out: List[Dict[str, float]] = []
    edge_idx = np.vstack([rows, cols])
    m_edges = int(len(w))
    ops_per_iter = _theoretical_ops_per_iter_flow_sparse(n, m_edges)
    rows_out.append(
        {
            "iter": 0.0,
            "runtime_sec": 0.0,
            "objective": np.nan,
            "constraint_l1": np.nan,
            "dual_l2_vs_lp": dual_l2_error(h, lp_dual),
            "dual_l2_debiased_vs_lp": (
                dual_l2_error(_w1_lipschitz_debias_potential(h, D), lp_dual)
                if D is not None
                else np.nan
            ),
            "dual_linf_vs_lp": dual_linf_error(h, lp_dual),
            "stable": 1.0,
            "ops_per_iter_theory": float(ops_per_iter),
            "ops_cum_theory": 0.0,
        }
    )

    eps = float(max(epsilon, 1e-12))
    relax = float(np.clip(relaxation, 1e-6, 1.0))
    for it in range(1, int(niter) + 1):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            loga_vals = -w / eps - h[cols] / eps
            logb_vals = -w / eps + h[cols] / eps
            loga = _segment_logsumexp_np(loga_vals, row_ptr, n)
            logb = _segment_logsumexp_np(logb_vals, row_ptr, n)

            if not np.all(np.isfinite(loga)) or not np.all(np.isfinite(logb)):
                break

            m = np.zeros(n, dtype=np.float64)
            if I0.size > 0:
                m[I0] = (loga[I0] - logb[I0]) / 2.0
            if Ip.size > 0:
                rp = np.maximum(r[Ip], 1e-300)
                log_ab = loga[Ip] + logb[Ip]
                # m = 0.5*(loga-logb) + asinh(r/sqrt(ab))
                log_ratio = np.log(rp) - 0.5 * log_ab
                m[Ip] = 0.5 * (loga[Ip] - logb[Ip]) + _asinh_exp_from_log(log_ratio)
            if In.size > 0:
                rn = np.maximum(-r[In], 1e-300)
                log_ab = loga[In] + logb[In]
                # m = 0.5*(loga-logb) - asinh((-r)/sqrt(ab))
                log_ratio = np.log(rn) - 0.5 * log_ab
                m[In] = 0.5 * (loga[In] - logb[In]) - _asinh_exp_from_log(log_ratio)

            h = (1.0 - relax) * h - relax * eps * m
            h = _center_in_place(h)
            if h_clip is not None:
                hc = float(abs(h_clip))
                h = np.clip(h, -hc, hc)
            logf = (-w + h[rows] - h[cols]) / eps
            f_vals = _safe_exp(logf)

        if not np.all(np.isfinite(f_vals)) or not np.all(np.isfinite(h)):
            crashed_iter = it
            break

        last_h = h.copy()
        last_f = f_vals.copy()
        last_iter = it

        if (it % max(1, int(record_every)) == 0) or (it == int(niter)):
            elapsed = time.perf_counter() - t_start
            obj = float(np.dot(f_vals, w))
            constraint_l1 = _divergence_residual_l1(edge_idx, f_vals, zf, n)
            rows_out.append(
                {
                    "iter": float(it),
                    "runtime_sec": float(elapsed),
                    "objective": obj,
                    "constraint_l1": constraint_l1,
                    "dual_l2_vs_lp": dual_l2_error(h, lp_dual),
                    "dual_l2_debiased_vs_lp": (
                        dual_l2_error(_w1_lipschitz_debias_potential(h, D), lp_dual)
                        if D is not None
                        else np.nan
                    ),
                    "dual_linf_vs_lp": dual_linf_error(h, lp_dual),
                    "stable": 1.0,
                    "ops_per_iter_theory": float(ops_per_iter),
                    "ops_cum_theory": float(it) * float(ops_per_iter),
                }
            )

    if crashed_iter > 0:
        rows_out.append(
            {
                "iter": float(crashed_iter),
                "runtime_sec": float(time.perf_counter() - t_start),
                "objective": np.nan,
                "constraint_l1": np.nan,
                "dual_l2_vs_lp": np.nan,
                "dual_l2_debiased_vs_lp": np.nan,
                "dual_linf_vs_lp": np.nan,
                "stable": 0.0,
                "ops_per_iter_theory": float(ops_per_iter),
                "ops_cum_theory": float(crashed_iter) * float(ops_per_iter),
            }
        )

    return {
        "trajectory": pd.DataFrame(rows_out),
        "h": last_h,
        "flow_values": last_f,
        "edge_indices": edge_idx,
        "completed_iters": int(last_iter),
        "crashed_iter": int(crashed_iter),
        "ops_per_iter_theory": float(ops_per_iter),
        "ops_total_theory": float(last_iter) * float(ops_per_iter),
    }


def _vanilla_sinkhorn_dense_iterations(
    distance_matrix: np.ndarray,
    z: np.ndarray,
    *,
    epsilon: float,
    niter: int,
    record_every: int,
    lp_dual: Optional[np.ndarray],
    h_clip: Optional[float] = 5000.0,
    relaxation: float = 0.5,
) -> Dict[str, object]:
    W = np.asarray(distance_matrix, dtype=np.float64)
    zf = z.astype(np.float64)
    n = len(zf)
    h = np.zeros(n, dtype=np.float64)
    last_h = h.copy()
    last_f = np.zeros((n, n), dtype=np.float64)
    last_iter = 0
    crashed_iter = 0
    r = zf / 2.0

    I0 = np.where(zf == 0)[0]
    Ip = np.where(zf > 0)[0]
    In = np.where(zf < 0)[0]

    t_start = time.perf_counter()
    rows_out: List[Dict[str, float]] = []
    eps = float(max(epsilon, 1e-12))
    relax = float(np.clip(relaxation, 1e-6, 1.0))
    ops_per_iter = _theoretical_ops_per_iter_vanilla_dense(n)
    rows_out.append(
        {
            "iter": 0.0,
            "runtime_sec": 0.0,
            "objective": np.nan,
            "constraint_l1": np.nan,
            "dual_l2_vs_lp": dual_l2_error(h, lp_dual),
            "dual_linf_vs_lp": dual_linf_error(h, lp_dual),
            "stable": 1.0,
            "ops_per_iter_theory": float(ops_per_iter),
            "ops_cum_theory": 0.0,
        }
    )

    for it in range(1, int(niter) + 1):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            loga = logsumexp(-W / eps + (-h / eps)[None, :], axis=1)
            logb = logsumexp(-W / eps + (+h / eps)[None, :], axis=1)
            if not np.all(np.isfinite(loga)) or not np.all(np.isfinite(logb)):
                break

            m = np.zeros(n, dtype=np.float64)
            if I0.size > 0:
                m[I0] = (loga[I0] - logb[I0]) / 2.0
            if Ip.size > 0:
                rp = np.maximum(r[Ip], 1e-300)
                log_ab = loga[Ip] + logb[Ip]
                log_ratio = np.log(rp) - 0.5 * log_ab
                m[Ip] = 0.5 * (loga[Ip] - logb[Ip]) + _asinh_exp_from_log(log_ratio)
            if In.size > 0:
                rn = np.maximum(-r[In], 1e-300)
                log_ab = loga[In] + logb[In]
                log_ratio = np.log(rn) - 0.5 * log_ab
                m[In] = 0.5 * (loga[In] - logb[In]) - _asinh_exp_from_log(log_ratio)

            h = (1.0 - relax) * h - relax * eps * m
            h = _center_in_place(h)
            if h_clip is not None:
                hc = float(abs(h_clip))
                h = np.clip(h, -hc, hc)
            logf = (-W + h[:, None] - h[None, :]) / eps
            f = _safe_exp(logf)

        if not np.all(np.isfinite(f)) or not np.all(np.isfinite(h)):
            crashed_iter = it
            break

        last_h = h.copy()
        last_f = f.copy()
        last_iter = it

        if (it % max(1, int(record_every)) == 0) or (it == int(niter)):
            elapsed = time.perf_counter() - t_start
            obj = float(np.sum(f * W))
            residual = float(np.linalg.norm((np.sum(f, axis=0) - np.sum(f, axis=1)) - zf, ord=1))
            rows_out.append(
                {
                    "iter": float(it),
                    "runtime_sec": float(elapsed),
                    "objective": obj,
                    "constraint_l1": residual,
                    "dual_l2_vs_lp": dual_l2_error(h, lp_dual),
                    "dual_linf_vs_lp": dual_linf_error(h, lp_dual),
                    "stable": 1.0,
                    "ops_per_iter_theory": float(ops_per_iter),
                    "ops_cum_theory": float(it) * float(ops_per_iter),
                }
            )

    if crashed_iter > 0:
        rows_out.append(
            {
                "iter": float(crashed_iter),
                "runtime_sec": float(time.perf_counter() - t_start),
                "objective": np.nan,
                "constraint_l1": np.nan,
                "dual_l2_vs_lp": np.nan,
                "dual_linf_vs_lp": np.nan,
                "stable": 0.0,
                "ops_per_iter_theory": float(ops_per_iter),
                "ops_cum_theory": float(crashed_iter) * float(ops_per_iter),
            }
        )

    return {
        "trajectory": pd.DataFrame(rows_out),
        "h": last_h,
        "flow": last_f,
        "completed_iters": int(last_iter),
        "crashed_iter": int(crashed_iter),
        "ops_per_iter_theory": float(ops_per_iter),
        "ops_total_theory": float(last_iter) * float(ops_per_iter),
    }


def run_flow_sinkhorn_sparse_trajectory(
    graph_indices: np.ndarray,
    graph_values: np.ndarray,
    graph_shape: Tuple[int, int],
    z: np.ndarray,
    *,
    epsilon: float,
    niter: int,
    lp_dual: Optional[np.ndarray] = None,
    distance_matrix: Optional[np.ndarray] = None,
    record_every: int = 1,
    h_clip: Optional[float] = 5000.0,
    relaxation: float = 0.8,
    use_torch: bool = False,
    device: Optional[str] = None,
    require_gpu: bool = False,
) -> pd.DataFrame:
    """Run sparse Flow-Sinkhorn and monitor per-iteration metrics.

    Returns a table with one row per recorded iteration.
    """
    if use_torch:
        out = _flow_sinkhorn_sparse_iterations_torch(
            graph_indices,
            graph_values,
            graph_shape,
            z,
            epsilon=epsilon,
            niter=niter,
            record_every=record_every,
            lp_dual=lp_dual,
            distance_matrix=distance_matrix,
            h_clip=h_clip,
            relaxation=relaxation,
            device=device,
            require_gpu=require_gpu,
        )
    else:
        out = _flow_sinkhorn_sparse_iterations(
            graph_indices,
            graph_values,
            graph_shape,
            z,
            epsilon=epsilon,
            niter=niter,
            record_every=record_every,
            lp_dual=lp_dual,
            distance_matrix=distance_matrix,
            h_clip=h_clip,
            relaxation=relaxation,
        )
    return out["trajectory"]


def run_vanilla_sinkhorn_dense_trajectory(
    distance_matrix: np.ndarray,
    z: np.ndarray,
    *,
    epsilon: float,
    niter: int,
    lp_dual: Optional[np.ndarray] = None,
    record_every: int = 1,
    h_clip: Optional[float] = 5000.0,
    relaxation: float = 0.5,
    use_torch: bool = False,
    device: Optional[str] = None,
    require_gpu: bool = False,
) -> pd.DataFrame:
    """Run dense Sinkhorn and monitor per-iteration metrics."""
    if use_torch:
        out = _vanilla_sinkhorn_dense_iterations_torch(
            distance_matrix,
            z,
            epsilon=epsilon,
            niter=niter,
            record_every=record_every,
            lp_dual=lp_dual,
            h_clip=h_clip,
            relaxation=relaxation,
            device=device,
            require_gpu=require_gpu,
        )
    else:
        out = _vanilla_sinkhorn_dense_iterations(
            distance_matrix,
            z,
            epsilon=epsilon,
            niter=niter,
            record_every=record_every,
            lp_dual=lp_dual,
            h_clip=h_clip,
            relaxation=relaxation,
        )
    return out["trajectory"]


def propose_epsilon_candidates(
    graph_values: np.ndarray,
    *,
    count: int = 14,
    exp_min: float = -3.5,
    exp_max: float = 0.5,
) -> np.ndarray:
    """Build log-spaced epsilon candidates from graph edge scales."""
    gv = np.asarray(graph_values, dtype=np.float64)
    base = float(np.median(gv))
    base = max(base, 1e-6)
    exps = np.linspace(exp_min, exp_max, count)
    eps = base * (10.0 ** exps)
    eps = np.unique(np.clip(eps, 1e-8, None))
    return eps


def screen_stable_epsilons(
    prepared: Dict[str, object],
    *,
    algorithm: str,
    eps_candidates: Sequence[float],
    warmup_iters: int = 20,
    max_keep: int = 9,
    check_iters: Optional[int] = None,
    distance_matrix: Optional[np.ndarray] = None,
) -> List[float]:
    """Keep epsilon values that remain finite and show usable progress."""
    mus = prepared["mus"]
    if len(mus) < 2:
        return []
    z = mus[0] - mus[-1]
    stable: List[float] = []

    ncheck = int(check_iters if check_iters is not None else warmup_iters)
    ncheck = max(ncheck, warmup_iters)
    z_l1 = float(np.linalg.norm(z, ord=1))

    D = distance_matrix
    if algorithm == "vanilla_sinkhorn_dense":
        if D is None:
            fw = floyd_warshall_metric(
                prepared["graph_indices"], prepared["graph_values"], prepared["graph_shape"]
            )
            D = fw["distances"]

    for eps in sorted(float(e) for e in eps_candidates):
        try:
            if algorithm == "flow_sinkhorn_sparse":
                df = run_flow_sinkhorn_sparse_trajectory(
                    prepared["graph_indices"],
                    prepared["graph_values"],
                    prepared["graph_shape"],
                    z,
                    epsilon=eps,
                    niter=ncheck,
                    lp_dual=None,
                    record_every=1,
                )
            elif algorithm == "vanilla_sinkhorn_dense":
                assert D is not None
                df = run_vanilla_sinkhorn_dense_trajectory(
                    D,
                    z,
                    epsilon=eps,
                    niter=ncheck,
                    lp_dual=None,
                    record_every=1,
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            if len(df) == 0:
                ok = False
            else:
                c = df["constraint_l1"].to_numpy(dtype=np.float64)
                o = df["objective"].to_numpy(dtype=np.float64)
                finite = np.isfinite(c).all() and np.isfinite(o).all()
                enough_len = len(df) >= max(6, ncheck // 3)
                start = float(c[0]) if len(c) > 0 else z_l1
                last = float(c[-1]) if len(c) > 0 else np.inf
                best = float(np.nanmin(c)) if len(c) > 0 else np.inf
                bounded = np.isfinite(last) and (last <= 1.25 * max(z_l1, 1e-12))
                improved = np.isfinite(start) and np.isfinite(best) and (best <= 0.95 * max(start, 1e-12))
                nontrivial = np.isfinite(best) and (best <= 0.82 * max(z_l1, 1e-12))
                ok = finite and enough_len and bounded and (improved or nontrivial)
        except Exception:
            ok = False
        if ok:
            stable.append(eps)

    if len(stable) <= max_keep:
        return stable

    idx = np.linspace(0, len(stable) - 1, max_keep, dtype=int)
    return [stable[i] for i in idx]


def _divergence_residual_l1(
    edge_indices: np.ndarray,
    edge_flows: np.ndarray,
    z: np.ndarray,
    n: int,
) -> float:
    rows = edge_indices[0]
    cols = edge_indices[1]
    div = np.zeros(n, dtype=np.float64)
    # Match solver convention: sum_in - sum_out = z
    np.add.at(div, cols, edge_flows)
    np.add.at(div, rows, -edge_flows)
    return float(np.linalg.norm(div - z, ord=1))


def benchmark_methods_on_prepared_data(
    prepared: Dict[str, object],
    *,
    epsilons: Optional[Sequence[float]] = None,
    epsilons_flow: Optional[Sequence[float]] = None,
    epsilons_vanilla: Optional[Sequence[float]] = None,
    niter: int = 300,
    use_torch: bool = True,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Benchmark Flow-Sinkhorn and vanilla Sinkhorn vs LP reference."""
    mus = prepared["mus"]
    if len(mus) < 2:
        raise ValueError("Need at least 2 time points.")
    z = mus[0] - mus[-1]

    graph_indices = prepared["graph_indices"]
    graph_values = prepared["graph_values"]
    graph_shape = prepared["graph_shape"]

    fw = floyd_warshall_metric(graph_indices, graph_values, graph_shape)
    D = fw["distances"]

    lp = solve_graph_w1_lp(graph_indices, graph_values, graph_shape, z)
    lp_obj = lp["objective"]
    lp_dual = lp["dual_potential"]

    rows: List[Dict[str, object]] = []
    rows.append(
        {
            "method": "floyd_warshall",
            "epsilon": np.nan,
            "objective": np.nan,
            "objective_rel_err_vs_lp": np.nan,
            "constraint_l1": np.nan,
            "dual_l2_vs_lp": np.nan,
            "dual_linf_vs_lp": np.nan,
            "time_sec": fw["time_sec"],
            "n_cells": prepared["n_cells"],
            "lp_time_sec": lp["time_sec"],
            "lp_objective": lp_obj,
        }
    )
    rows.append(
        {
            "method": "lp_highs",
            "epsilon": np.nan,
            "objective": lp_obj,
            "objective_rel_err_vs_lp": 0.0,
            "constraint_l1": 0.0,
            "dual_l2_vs_lp": 0.0,
            "dual_linf_vs_lp": 0.0,
            "time_sec": lp["time_sec"],
            "n_cells": prepared["n_cells"],
            "lp_time_sec": lp["time_sec"],
            "lp_objective": lp_obj,
        }
    )

    if epsilons is not None:
        flow_eps = [float(e) for e in epsilons]
        vanilla_eps = [float(e) for e in epsilons]
    else:
        flow_eps = [float(e) for e in (epsilons_flow or [])]
        vanilla_eps = [float(e) for e in (epsilons_vanilla or [])]

    for eps in flow_eps:
        flow = solve_flow_sinkhorn_sparse(
            graph_indices,
            graph_values,
            graph_shape,
            z,
            epsilon=float(eps),
            niter=niter,
            use_torch=use_torch,
            device=device,
        )
        rows.append(
            {
                "method": "flow_sinkhorn_sparse",
                "epsilon": float(eps),
                "objective": flow["objective"],
                "objective_rel_err_vs_lp": abs(flow["objective"] - lp_obj) / max(abs(lp_obj), 1e-12),
                "constraint_l1": flow["constraint_l1"],
                "dual_l2_vs_lp": dual_l2_error(flow["h"], lp_dual),
                "dual_linf_vs_lp": dual_linf_error(flow["h"], lp_dual),
                "time_sec": flow["time_sec"],
                "n_cells": prepared["n_cells"],
                "lp_time_sec": lp["time_sec"],
                "lp_objective": lp_obj,
            }
        )

    for eps in vanilla_eps:
        vanilla = solve_vanilla_sinkhorn_dense(
            D,
            z,
            epsilon=float(eps),
            niter=niter,
        )
        rows.append(
            {
                "method": "vanilla_sinkhorn_dense",
                "epsilon": float(eps),
                "objective": vanilla["objective"],
                "objective_rel_err_vs_lp": abs(vanilla["objective"] - lp_obj) / max(abs(lp_obj), 1e-12),
                "constraint_l1": vanilla["constraint_l1"],
                "dual_l2_vs_lp": dual_l2_error(vanilla["h"], lp_dual),
                "dual_linf_vs_lp": dual_linf_error(vanilla["h"], lp_dual),
                "time_sec": vanilla["time_sec"],
                "n_cells": prepared["n_cells"],
                "lp_time_sec": lp["time_sec"],
                "lp_objective": lp_obj,
            }
        )

    return pd.DataFrame(rows)


def reconstruct_shortest_path(
    predecessors: np.ndarray,
    i: int,
    j: int,
) -> List[int]:
    """Reconstruct shortest path from i to j using Floyd-Warshall predecessors."""
    if i == j:
        return [i]
    if predecessors[i, j] < 0:
        return []
    path = [j]
    cur = j
    while cur != i:
        cur = int(predecessors[i, cur])
        if cur < 0:
            return []
        path.append(cur)
        if len(path) > predecessors.shape[0] + 5:
            return []
    path.reverse()
    return path


def sinkhorn_coupling(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    *,
    epsilon: float,
    niter: int = 400,
    tol: float = 1e-9,
) -> np.ndarray:
    """Balanced entropic OT coupling via classical Sinkhorn scaling."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    K = np.exp(-C / max(epsilon, 1e-12))
    K = np.maximum(K, 1e-300)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(niter):
        u_prev = u.copy()
        Ku = K @ v
        u = a / np.maximum(Ku, 1e-300)
        Kv = K.T @ u
        v = b / np.maximum(Kv, 1e-300)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    P = (u[:, None] * K) * v[None, :]
    total = P.sum()
    if total > 0:
        P /= total
    return P


def interpolate_from_endpoint_coupling(
    prepared: Dict[str, object],
    *,
    epsilon: float = 0.05,
    sinkhorn_niter: int = 400,
    mass_threshold: float = 1e-6,
) -> Dict[str, object]:
    """Interpolate mu_1->mu_T on graph shortest paths and compare to real days."""
    days = prepared["days"]
    if len(days) < 2:
        raise ValueError("Need at least two time points.")

    day_to_indices = prepared["day_to_indices"]
    idx_src = day_to_indices[float(days[0])]
    idx_tgt = day_to_indices[float(days[-1])]
    n = prepared["n_cells"]

    fw = floyd_warshall_metric(
        prepared["graph_indices"],
        prepared["graph_values"],
        prepared["graph_shape"],
        return_predecessors=True,
    )
    D = fw["distances"]
    pred = fw["predecessors"]

    C = D[np.ix_(idx_src, idx_tgt)]
    a = np.ones(len(idx_src), dtype=np.float64) / len(idx_src)
    b = np.ones(len(idx_tgt), dtype=np.float64) / len(idx_tgt)
    P = sinkhorn_coupling(a, b, C, epsilon=epsilon, niter=sinkhorn_niter)

    edge_lookup = {}
    gi = prepared["graph_indices"]
    gv = prepared["graph_values"]
    for e in range(gv.size):
        edge_lookup[(int(gi[0, e]), int(gi[1, e]))] = float(gv[e])

    day0 = float(days[0])
    dayT = float(days[-1])
    interp = {}
    gt = {}

    for d in days:
        alpha = 0.0 if dayT == day0 else (float(d) - day0) / (dayT - day0)
        mass = np.zeros(n, dtype=np.float64)
        for ii, i_node in enumerate(idx_src):
            for jj, j_node in enumerate(idx_tgt):
                m = P[ii, jj]
                if m < mass_threshold:
                    continue
                path = reconstruct_shortest_path(pred, int(i_node), int(j_node))
                if not path:
                    continue
                if len(path) == 1:
                    mass[path[0]] += m
                    continue
                seg_lens = []
                for u, v in zip(path[:-1], path[1:]):
                    w = edge_lookup.get((u, v), edge_lookup.get((v, u), 1.0))
                    seg_lens.append(float(w))
                cum = np.cumsum(seg_lens)
                total = float(cum[-1])
                target = alpha * total
                k = int(np.searchsorted(cum, target, side="left"))
                k = min(k, len(path) - 1)
                mass[path[k]] += m
        if mass.sum() > 0:
            mass /= mass.sum()
        interp[float(d)] = mass

        gt_mass = np.zeros(n, dtype=np.float64)
        gt_idx = day_to_indices[float(d)]
        gt_mass[gt_idx] = 1.0 / max(len(gt_idx), 1)
        gt[float(d)] = gt_mass

    tv_by_day = {
        d: 0.5 * float(np.sum(np.abs(interp[d] - gt[d])))
        for d in interp
    }
    return {
        "interp_mass_by_day": interp,
        "ground_truth_mass_by_day": gt,
        "tv_by_day": tv_by_day,
        "endpoint_coupling": P,
        "floyd_time_sec": fw["time_sec"],
    }
