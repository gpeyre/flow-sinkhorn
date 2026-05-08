#!/usr/bin/env python3
"""Unified benchmark runner for line, single-cell, and Delaunay graphs.

Outputs:
- neurips/figures/benchmark-<setup>-<bench>-{graph,convergence}.pdf
- benchmarks/results/tables/report-<bench>-*.csv|tex
- benchmarks/results/report-<bench>.pdf
- benchmarks/results/figures/report-<bench>-*.pdf (detailed diagnostics)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.wot_benchmark import (
    BenchmarkConfig,
    floyd_warshall_metric,
    prepare_wot_data,
    propose_epsilon_candidates,
    solve_flow_sinkhorn_sparse,
    run_flow_sinkhorn_sparse_trajectory,
    run_vanilla_sinkhorn_dense_trajectory,
    screen_stable_epsilons,
    solve_graph_w1_lp,
    solve_line_w1_dual_closed_form,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench", choices=["line", "singlecell", "delaunay"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=240, help="Total graph size for line/delaunay or target total cells for singlecell")
    p.add_argument("--n0", type=int, default=60, help="Cells per time snapshot for singlecell")
    p.add_argument("--k", type=int, default=4, help="k for singlecell kNN graph")
    p.add_argument("--pca", type=int, default=30, help="PCA dimension for singlecell")
    p.add_argument("--m", type=int, default=None, help="Support size per measure (line/delaunay). Default n//20")
    p.add_argument("--flow-eps-count", "--flow-gamma-count", dest="flow_eps_count", type=int, default=8)
    p.add_argument("--vanilla-eps-count", "--vanilla-gamma-count", dest="vanilla_eps_count", type=int, default=8)
    p.add_argument("--flow-gammas", type=float, nargs="+", default=None)
    p.add_argument("--vanilla-gammas", type=float, nargs="+", default=None)
    p.add_argument("--flow-base-iters", type=int, default=900)
    p.add_argument("--flow-max-iters", type=int, default=7000)
    p.add_argument("--vanilla-base-iters", type=int, default=1200)
    p.add_argument("--vanilla-max-iters", type=int, default=12000)
    p.add_argument("--out-root", type=Path, default=Path("neurips"))
    p.add_argument("--setup-tag", type=str, default="small", help="Tag for paper panel naming, e.g. small/large.")
    p.add_argument("--data-root", type=Path, default=Path("data/wot"))
    p.add_argument("--full", action="store_true", help="Use a broader gamma sweep and longer runtimes.")
    p.add_argument("--use-torch", action="store_true", help="Use PyTorch backend for trajectory benchmarks.")
    p.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda or cpu.")
    p.add_argument("--require-gpu", action="store_true", help="Fail if CUDA GPU backend is not active.")
    p.add_argument(
        "--line-closed-form-w1",
        action="store_true",
        help=(
            "For line benchmark, use closed-form unregularized W1 as reference objective. "
            "This also switches the geodesic distance matrix to analytic path distance |i-j|."
        ),
    )
    p.add_argument(
        "--line-analytic-distance",
        action="store_true",
        help="For line benchmark, use analytic path distance |i-j| instead of Floyd-Warshall.",
    )
    p.add_argument(
        "--line-check-closed-form",
        action="store_true",
        help="For line benchmark, assert machine-precision agreement for distance and W1 objective checks.",
    )
    return p.parse_args()


def _path_graph(n: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray]:
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
        vals.extend([1.0, 1.0])
    idx = np.vstack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)])
    w = np.asarray(vals, dtype=np.float64)
    pts = np.column_stack([np.arange(n, dtype=np.float64), np.zeros(n, dtype=np.float64)])
    return idx, w, (n, n), pts


def _delaunay_graph(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    tri = Delaunay(pts)
    undirected = set()
    for t in tri.simplices:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for i, j in [(a, b), (b, c), (c, a)]:
            u, v = (i, j) if i < j else (j, i)
            undirected.add((u, v))

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for u, v in sorted(undirected):
        d = float(np.linalg.norm(pts[u] - pts[v]))
        rows.extend([u, v])
        cols.extend([v, u])
        vals.extend([d, d])
    idx = np.vstack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)])
    w = np.asarray(vals, dtype=np.float64)
    return idx, w, (n, n), pts


def _endpoint_measures(n: int, m: int, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(score)
    left = order[:m]
    right = order[-m:]
    mu0 = np.zeros(n, dtype=np.float64)
    mu1 = np.zeros(n, dtype=np.float64)
    mu0[left] = 1.0 / m
    mu1[right] = 1.0 / m
    return mu0, mu1, (mu0 - mu1)


def _corner_measures(points: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source on top-left, target on bottom-right, with disjoint supports."""
    n = int(points.shape[0])
    p = np.asarray(points, dtype=np.float64)
    m = int(min(max(2, m), max(2, n // 2 - 1)))

    d_src = np.linalg.norm(p - np.array([0.0, 1.0], dtype=np.float64), axis=1)
    d_tgt = np.linalg.norm(p - np.array([1.0, 0.0], dtype=np.float64), axis=1)

    src = np.argsort(d_src)[:m]
    avail = np.setdiff1d(np.arange(n, dtype=np.int64), src, assume_unique=False)
    tgt_rank = avail[np.argsort(d_tgt[avail])]
    tgt = tgt_rank[:m]

    mu0 = np.zeros(n, dtype=np.float64)
    mu1 = np.zeros(n, dtype=np.float64)
    mu0[src] = 1.0 / m
    mu1[tgt] = 1.0 / m
    return mu0, mu1, mu0 - mu1


def prepare_problem(args: argparse.Namespace) -> Dict[str, object]:
    if args.bench == "line":
        n = int(args.n)
        m = int(args.m if args.m is not None else max(2, n // 20))
        gi, gv, gs, pts = _path_graph(n)
        mu0, mu1, z = _endpoint_measures(n, m, score=pts[:, 0])
        return {
            "name": "line",
            "graph_indices": gi,
            "graph_values": gv,
            "graph_shape": gs,
            "mus": [mu0, mu1],
            "z": z,
            "points": pts,
            "meta": {"n": n, "m": m},
        }

    if args.bench == "delaunay":
        n = int(args.n)
        m = int(args.m if args.m is not None else max(2, n // 20))
        gi, gv, gs, pts = _delaunay_graph(n=n, seed=args.seed)
        mu0, mu1, z = _corner_measures(pts, m)
        return {
            "name": "delaunay",
            "graph_indices": gi,
            "graph_values": gv,
            "graph_shape": gs,
            "mus": [mu0, mu1],
            "z": z,
            "points": pts,
            "meta": {"n": n, "m": m},
        }

    # singlecell
    n_total = int(args.n)
    n0 = int(args.n0)
    max_t = max(2, int(np.floor(n_total / max(n0, 1))))
    cfg = BenchmarkConfig(
        data_root=args.data_root,
        random_state=args.seed,
        n_cells_per_time=n0,
        pca_components=args.pca,
        knn_k=args.k,
        max_timepoints=max_t,
        use_torch=False,
    )
    prepared = prepare_wot_data(cfg)
    z = np.asarray(prepared["mus"][0] - prepared["mus"][-1], dtype=np.float64)
    return {
        "name": "singlecell",
        "graph_indices": prepared["graph_indices"],
        "graph_values": prepared["graph_values"],
        "graph_shape": prepared["graph_shape"],
        "mus": [prepared["mus"][0], prepared["mus"][-1]],
        "z": z,
        "points": np.asarray(prepared["embedding"])[:, :2],
        "days": np.asarray(prepared["sampled_days"]),
        "meta": {
            "n": int(prepared["graph_shape"][0]),
            "n0": n0,
            "k": int(args.k),
            "n_time": int(len(prepared["days"])),
        },
    }


def _budget(eps: float, eps_ref: float, base: int, max_iter: int, power: float) -> int:
    scale = (eps_ref / max(float(eps), 1e-12)) ** power
    out = int(np.ceil(base * scale))
    return int(min(max_iter, max(base, out)))


def _best_so_far(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    out = np.full_like(y, np.nan)
    b = np.inf
    for i, v in enumerate(y):
        if np.isfinite(v):
            b = min(b, float(v))
            out[i] = b
    return out


def _tail_rel_drop(best_curve: np.ndarray, frac: float = 0.1) -> float:
    y = np.asarray(best_curve, dtype=np.float64)
    y = y[np.isfinite(y) & (y > 0)]
    if y.size < 20:
        return np.inf
    i0 = int(max(0, np.floor((1.0 - frac) * y.size)))
    a = float(y[i0])
    b = float(y[-1])
    return float((a - b) / max(a, 1e-12))


def _select_gamma_grid(graph_values: np.ndarray, args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    gv = np.asarray(graph_values, dtype=np.float64)
    base = max(float(np.median(gv)), 1e-6)

    n_flow = int(args.flow_eps_count + (5 if args.full else 0))
    n_van = int(args.vanilla_eps_count + (5 if args.full else 0))

    # Broaden Flow range to include larger gamma regimes explicitly.
    flow = np.geomspace(base * 1e-4, base * 3e1, n_flow)
    vanilla = np.geomspace(base * 8e-4, base * 6e1, n_van)
    return sorted(float(e) for e in np.unique(flow)), sorted(float(e) for e in np.unique(vanilla))


def _run_with_plateau_flow(
    gi: np.ndarray,
    gv: np.ndarray,
    gs: Tuple[int, int],
    z: np.ndarray,
    lp_dual: np.ndarray,
    lp_norm: float,
    gamma: float,
    base_iters: int,
    max_iters: int,
    use_torch: bool = False,
    device: str | None = None,
    require_gpu: bool = False,
) -> Tuple[pd.DataFrame, int, float]:
    niter = int(base_iters)
    last = pd.DataFrame()
    tail_drop = np.inf
    while True:
        tr = run_flow_sinkhorn_sparse_trajectory(
            gi, gv, gs, z,
            epsilon=float(gamma),
            niter=int(niter),
            lp_dual=lp_dual,
            record_every=1,
            h_clip=None,
            relaxation=0.8,
            use_torch=use_torch,
            device=device,
            require_gpu=require_gpu,
        )
        tr["dual_l2_rel_vs_lp"] = tr["dual_l2_vs_lp"] / lp_norm
        y = _best_so_far(tr["dual_l2_rel_vs_lp"].to_numpy(dtype=np.float64))
        tail_drop = _tail_rel_drop(y, frac=0.12)
        last = tr
        if (tail_drop <= 8e-4) or (niter >= max_iters):
            return last, int(niter), float(tail_drop)
        niter = int(min(max_iters, int(np.ceil(1.8 * niter))))


def _run_with_plateau_vanilla(
    D: np.ndarray,
    z: np.ndarray,
    lp_dual: np.ndarray,
    lp_norm: float,
    gamma: float,
    base_iters: int,
    max_iters: int,
    use_torch: bool = False,
    device: str | None = None,
    require_gpu: bool = False,
) -> Tuple[pd.DataFrame, int, float]:
    niter = int(base_iters)
    last = pd.DataFrame()
    tail_drop = np.inf
    while True:
        tr = run_vanilla_sinkhorn_dense_trajectory(
            D,
            z,
            epsilon=float(gamma),
            niter=int(niter),
            lp_dual=lp_dual,
            record_every=1,
            relaxation=0.5,
            use_torch=use_torch,
            device=device,
            require_gpu=require_gpu,
        )
        tr["dual_l2_rel_vs_lp"] = tr["dual_l2_vs_lp"] / lp_norm
        y = _best_so_far(tr["dual_l2_rel_vs_lp"].to_numpy(dtype=np.float64))
        tail_drop = _tail_rel_drop(y, frac=0.12)
        last = tr
        # stricter plateau requirement for vanilla as requested.
        if (tail_drop <= 4e-4) or (niter >= max_iters):
            return last, int(niter), float(tail_drop)
        niter = int(min(max_iters, int(np.ceil(1.9 * niter))))


def _pick_display_gammas(gammas: List[float], k: int = 4) -> List[float]:
    vals = sorted(float(e) for e in gammas)
    if len(vals) <= k:
        return vals
    idx = np.linspace(0, len(vals) - 1, k, dtype=int)
    return [vals[i] for i in idx]


def _pick_gammas_from_final_error(summary: pd.DataFrame, method: str, k: int) -> List[float]:
    ss = summary[summary["method"] == method].copy()
    if len(ss) == 0:
        return []
    ss = ss.sort_values("best_final_dual_l2_rel")
    vals = ss["gamma"].to_numpy(dtype=np.float64)
    if vals.size <= k:
        return [float(v) for v in vals]
    idx = np.linspace(0, vals.size - 1, k, dtype=int)
    return [float(vals[i]) for i in idx]


def _net_undirected_flow(
    edge_indices: np.ndarray,
    flow_values: np.ndarray,
) -> Dict[Tuple[int, int], float]:
    signed: Dict[Tuple[int, int], float] = {}
    eidx = np.asarray(edge_indices, dtype=np.int64)
    fv = np.asarray(flow_values, dtype=np.float64)
    for k in range(eidx.shape[1]):
        i = int(eidx[0, k])
        j = int(eidx[1, k])
        if i == j:
            continue
        u, v = (i, j) if i < j else (j, i)
        sgn = 1.0 if (i < j) else -1.0
        signed[(u, v)] = signed.get((u, v), 0.0) + sgn * float(fv[k])
    return {edge: abs(val) for edge, val in signed.items()}


def _line_path_distance_matrix(n: int) -> np.ndarray:
    idx = np.arange(int(n), dtype=np.float64)
    return np.abs(idx[:, None] - idx[None, :])


def _line_w1_closed_form(z: np.ndarray) -> float:
    zz = np.asarray(z, dtype=np.float64).ravel()
    c = np.cumsum(zz)
    if c.size <= 1:
        return 0.0
    return float(np.sum(np.abs(c[:-1])))


def run_benchmark(problem: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    gi = np.asarray(problem["graph_indices"]) 
    gv = np.asarray(problem["graph_values"], dtype=np.float64)
    gs = tuple(problem["graph_shape"])
    z = np.asarray(problem["z"], dtype=np.float64)
    bench_name = str(problem.get("name", ""))
    n_nodes = int(gs[0])

    lp = solve_graph_w1_lp(gi, gv, gs, z)
    line_w1_ref = _line_w1_closed_form(z) if bench_name == "line" else np.nan
    if bench_name == "line" and args.line_closed_form_w1:
        lp["objective"] = float(line_w1_ref)
        lp_cf = solve_line_w1_dual_closed_form(z, gauge="centered")
        lp["dual_potential"] = np.asarray(lp_cf["dual_potential"], dtype=np.float64)

    if bench_name == "line" and args.line_check_closed_form:
        D_fw_check = floyd_warshall_metric(gi, gv, gs)["distances"]
        D_path_check = _line_path_distance_matrix(n_nodes)
        d_err = float(np.max(np.abs(D_fw_check - D_path_check)))
        if d_err > 1e-12:
            raise RuntimeError(
                f"Line distance sanity check failed: max|D_fw - D_path|={d_err:.3e} > 1e-12"
            )
        lp_check = solve_graph_w1_lp(gi, gv, gs, z)
        lp_obj_raw = float(lp_check["objective"])
        w1_err = abs(float(line_w1_ref) - lp_obj_raw)
        if w1_err > 1e-10:
            raise RuntimeError(
                f"Line W1 sanity check failed: |W1_closed - W1_lp|={w1_err:.3e} > 1e-10"
            )
        h_cf = solve_line_w1_dual_closed_form(z, gauge="centered")["dual_potential"]
        h_lp = np.asarray(lp_check["dual_potential"], dtype=np.float64)
        h_lp = h_lp - float(np.mean(h_lp))
        dual_err = float(np.linalg.norm(h_cf - h_lp, ord=2))
        if dual_err > 1e-8:
            raise RuntimeError(
                f"Line dual sanity check failed: ||h_closed - h_lp||_2={dual_err:.3e} > 1e-8"
            )
        print(f"[check] line_distance_max_abs_err={d_err:.3e}")
        print(f"[check] line_w1_abs_err={w1_err:.3e}")
        print(f"[check] line_dual_l2_err={dual_err:.3e}")

    lp_dual = np.asarray(lp["dual_potential"], dtype=np.float64)
    lp_dual = lp_dual - float(np.mean(lp_dual))
    lp_norm = max(float(np.linalg.norm(lp_dual)), 1e-12)

    flow_candidates, van_candidates = _select_gamma_grid(gv, args)
    if bench_name == "delaunay":
        # Keep vanilla gamma in a tighter, low-regularization regime.
        van_candidates = [float(e) for e in van_candidates if float(e) <= 1.05e-2]
    elif bench_name == "singlecell":
        # Requested single-cell vanilla gamma range.
        van_candidates = [float(e) for e in van_candidates if 1e-1 <= float(e) <= 5.0]
    prepared_for_screen = {
        "mus": problem["mus"],
        "graph_indices": gi,
        "graph_values": gv,
        "graph_shape": gs,
    }
    if bench_name == "line" and (args.line_analytic_distance or args.line_closed_form_w1):
        D = _line_path_distance_matrix(n_nodes)
    else:
        D = floyd_warshall_metric(gi, gv, gs)["distances"]
    if args.flow_gammas is not None:
        flow_eps = [float(g) for g in args.flow_gammas]
    else:
        flow_stable = screen_stable_epsilons(
            prepared_for_screen,
            algorithm="flow_sinkhorn_sparse",
            eps_candidates=flow_candidates,
            warmup_iters=30,
            max_keep=max(6, int(args.flow_eps_count) + 4),
        )
        flow_eps = _pick_display_gammas(flow_stable if len(flow_stable) > 0 else flow_candidates, k=max(4, int(args.flow_eps_count)))
    if args.vanilla_gammas is not None:
        van_eps = [float(g) for g in args.vanilla_gammas]
    else:
        if bench_name == "singlecell":
            # Keep a visible multi-gamma vanilla family in the requested range.
            n_v = max(4, int(args.vanilla_eps_count))
            van_eps = [float(x) for x in np.geomspace(1e-1, 5.0, n_v)]
        elif bench_name == "delaunay":
            n_v = max(4, int(args.vanilla_eps_count))
            van_eps = [float(x) for x in np.geomspace(8.9e-4, 1.0e-2, n_v)]
        else:
            van_stable = screen_stable_epsilons(
                prepared_for_screen,
                algorithm="vanilla_sinkhorn_dense",
                eps_candidates=van_candidates,
                warmup_iters=30,
                max_keep=max(6, int(args.vanilla_eps_count) + 4),
                distance_matrix=D,
            )
            van_eps = _pick_display_gammas(van_stable if len(van_stable) > 0 else van_candidates, k=max(4, int(args.vanilla_eps_count)))
    if bench_name == "singlecell":
        # Keep the user-requested vanilla range and let stability screening happen naturally.
        van_eps = [float(e) for e in van_eps if 1e-1 <= float(e) <= 5.0]
    if bench_name == "delaunay":
        van_eps = [float(e) for e in van_eps if float(e) <= 1.05e-2]
    if len(van_eps) == 0:
        if bench_name == "singlecell":
            van_eps = [0.1, 0.25, 0.75, 2.0, 5.0]
        elif bench_name == "delaunay":
            van_eps = [0.001, 0.003, 0.006, 0.0095]
    flow_eps = sorted(float(e) for e in np.unique(flow_eps))
    van_eps = sorted(float(e) for e in np.unique(van_eps))

    # Same iteration budget across gammas for each algorithm.
    flow_niter = int(args.flow_max_iters)
    vanilla_niter = int(args.vanilla_max_iters)

    rows: List[pd.DataFrame] = []

    for i, eps in enumerate(flow_eps, start=1):
        print(f"Flow Sinkhorn bench: {i}/{len(flow_eps)} (gamma={eps})")
        tr = run_flow_sinkhorn_sparse_trajectory(
            gi,
            gv,
            gs,
            z,
            epsilon=float(eps),
            niter=flow_niter,
            lp_dual=lp_dual,
            distance_matrix=D,
            record_every=1,
            h_clip=None,
            relaxation=0.8,
            use_torch=args.use_torch,
            device=args.device,
            require_gpu=args.require_gpu,
        )
        tr["method"] = "flow_sinkhorn_sparse"
        tr["epsilon"] = float(eps)
        tr["gamma"] = float(eps)
        tr["niter_budget"] = int(flow_niter)
        tr["tail_rel_drop_last12pct"] = float(
            _tail_rel_drop(_best_so_far(tr["dual_l2_vs_lp"].to_numpy(dtype=np.float64) / lp_norm), frac=0.12)
        )
        rows.append(tr)

    for i, eps in enumerate(van_eps, start=1):
        print(f"Vanilla Sinkhorn bench: {i}/{len(van_eps)} (gamma={eps})")
        tr = run_vanilla_sinkhorn_dense_trajectory(
            D,
            z,
            epsilon=float(eps),
            niter=vanilla_niter,
            lp_dual=lp_dual,
            record_every=1,
            h_clip=None,
            relaxation=0.5,
            use_torch=args.use_torch,
            device=args.device,
            require_gpu=args.require_gpu,
        )
        tr["method"] = "vanilla_sinkhorn_dense"
        tr["epsilon"] = float(eps)
        tr["gamma"] = float(eps)
        tr["niter_budget"] = int(vanilla_niter)
        tr["tail_rel_drop_last12pct"] = float(
            _tail_rel_drop(_best_so_far(tr["dual_l2_vs_lp"].to_numpy(dtype=np.float64) / lp_norm), frac=0.12)
        )
        rows.append(tr)

    curves = pd.concat(rows, ignore_index=True)
    curves["dual_l2_rel_vs_lp"] = curves["dual_l2_vs_lp"] / lp_norm
    if "dual_l2_debiased_vs_lp" not in curves.columns:
        curves["dual_l2_debiased_vs_lp"] = np.nan
    curves["dual_l2_debiased_rel_vs_lp"] = curves["dual_l2_debiased_vs_lp"] / lp_norm

    summary_rows = []
    for (method, gamma), ss in curves.groupby(["method", "gamma"]):
        ss = ss.sort_values("iter")
        y = _best_so_far(ss["dual_l2_rel_vs_lp"].to_numpy(dtype=np.float64))
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        summary_rows.append(
            {
                "method": method,
                "epsilon": float(gamma),
                "gamma": float(gamma),
                "best_final_dual_l2_rel": float(y[-1]),
                "n_points": int(y.size),
                "runtime_sec": float(ss["runtime_sec"].max()),
                "tail_rel_drop_last12pct": float(_tail_rel_drop(y, frac=0.12)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["method", "gamma"]) if summary_rows else pd.DataFrame()

    # Redistribute gammas from achieved final errors (proxy for target unregularized accuracy).
    flow_eps = _pick_gammas_from_final_error(summary, "flow_sinkhorn_sparse", k=max(4, int(args.flow_eps_count)))
    van_eps = _pick_gammas_from_final_error(summary, "vanilla_sinkhorn_dense", k=max(4, int(args.vanilla_eps_count)))

    # Converged flow visualizations for a subset of epsilons.
    flow_display_eps = _pick_display_gammas(flow_eps, k=4)
    flow_viz = []
    for eps in flow_display_eps:
        row = summary[(summary["method"] == "flow_sinkhorn_sparse") & (summary["gamma"] == eps)]
        niter_used = int(row["n_points"].iloc[0]) if len(row) > 0 else int(args.flow_max_iters)
        solved = solve_flow_sinkhorn_sparse(
            gi, gv, gs, z,
            epsilon=float(eps),
            niter=max(100, niter_used),
            use_torch=args.use_torch,
            device=args.device,
            h_clip=None,
            relaxation=0.8,
        )
        flow_viz.append(
            {
                "epsilon": float(eps),
                "gamma": float(eps),
                "edge_indices": np.asarray(solved["edge_indices"], dtype=np.int64),
                "flow_values": np.asarray(solved["flow_values"], dtype=np.float64),
            }
        )

    # Dedicated flow display with smaller gamma and longer optimization until plateau.
    medium_flow = None
    if len(flow_eps) > 0:
        gamma_small = float(sorted(flow_eps)[max(0, len(flow_eps) // 3 - 1)])
        tr_med, niter_med, _ = _run_with_plateau_flow(
            gi, gv, gs, z, lp_dual, lp_norm, gamma=gamma_small,
            base_iters=max(1400, args.flow_base_iters),
            max_iters=max(6000, args.flow_max_iters * 4),
            use_torch=args.use_torch,
            device=args.device,
            require_gpu=args.require_gpu,
        )
        solved_med = solve_flow_sinkhorn_sparse(
            gi, gv, gs, z,
            epsilon=gamma_small,
            niter=max(500, int(niter_med)),
            use_torch=args.use_torch,
            device=args.device,
            h_clip=None,
            relaxation=0.8,
        )
        medium_flow = {
            "gamma": gamma_small,
            "edge_indices": np.asarray(solved_med["edge_indices"], dtype=np.int64),
            "flow_values": np.asarray(solved_med["flow_values"], dtype=np.float64),
            "plateau_niter": int(niter_med),
            "plateau_runtime_sec": float(tr_med["runtime_sec"].max()) if len(tr_med) > 0 else np.nan,
        }

    return {
        "curves": curves,
        "summary": summary,
        "lp_objective": float(lp["objective"]),
        "line_w1_closed_form": float(line_w1_ref) if bench_name == "line" else np.nan,
        "flow_eps": flow_eps,
        "vanilla_eps": van_eps,
        "flow_viz": flow_viz,
        "medium_flow": medium_flow,
        "distance_matrix_time_sec": np.nan,
        "problem": problem,
    }


def _plot_problem(problem: Dict[str, object], out_pdf: Path) -> None:
    pts = np.asarray(problem["points"], dtype=np.float64)
    mu0 = np.asarray(problem["mus"][0], dtype=np.float64)
    mu1 = np.asarray(problem["mus"][1], dtype=np.float64)
    gi = np.asarray(problem["graph_indices"], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    # Thin graph edges
    for i, j in zip(gi[0], gi[1]):
        if i < j:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#C8CCD2", lw=0.5, zorder=1)

    ax.scatter(pts[:, 0], pts[:, 1], s=9, color="#A9B2BC", alpha=0.75, zorder=2)
    src = np.where(mu0 > 0)[0]
    tgt = np.where(mu1 > 0)[0]
    ax.scatter(pts[src, 0], pts[src, 1], s=38, color="#1f77b4", label="source support", zorder=3)
    ax.scatter(pts[tgt, 0], pts[tgt, 1], s=38, color="#d62728", label="target support", zorder=3)
    ax.grid(alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _eps_color(i: int, n: int) -> Tuple[float, float, float]:
    if n <= 1:
        return (0.5, 0.0, 0.5)
    t = float(i) / float(n - 1)
    return (t, 0.0, 1.0 - t)


def _plot_error_decay(
    curves: pd.DataFrame,
    method: str,
    out_pdf: Path,
    title: str,
    x_max: float,
    y_limits: Tuple[float, float] | None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    sub = curves[curves["method"] == method].copy()
    eps_vals = sorted(sub["gamma"].dropna().unique())
    for i, eps in enumerate(eps_vals):
        ss = sub[sub["gamma"] == eps].sort_values("runtime_sec")
        x = ss["runtime_sec"].to_numpy(dtype=np.float64)
        y = ss["dual_l2_rel_vs_lp"].to_numpy(dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y > 0)
        if np.sum(good) < 2:
            continue
        ax.plot(x[good], _best_so_far(y[good]), lw=1.8, color=_eps_color(i, len(eps_vals)), label=f"{eps:g}")
        if method == "flow_sinkhorn_sparse":
            yd = ss["dual_l2_debiased_rel_vs_lp"].to_numpy(dtype=np.float64)
            good_d = np.isfinite(x) & np.isfinite(yd) & (x >= 0) & (yd > 0)
            if np.sum(good_d) >= 2:
                ax.plot(x[good_d], _best_so_far(yd[good_d]), lw=1.5, ls="--", color=_eps_color(i, len(eps_vals)))
    ax.set_title(title)
    ax.set_xlabel("wall clock time (sec)")
    ax.set_ylabel("best-so-far relative centered dual L2 error")
    ax.set_yscale("log")
    if np.isfinite(x_max) and x_max > 0:
        ax.set_xlim(0.0, x_max)
    if y_limits is not None and all(np.isfinite(y_limits)) and y_limits[0] > 0 and y_limits[1] > y_limits[0]:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=0.25)
    gamma_legend = ax.legend(title=r"$\gamma$", fontsize=8, ncol=2, loc="best")
    if method == "flow_sinkhorn_sparse":
        ax.add_artist(gamma_legend)
        style_handles = [
            Line2D([0], [0], color="#333333", lw=1.8, ls="-", label="original"),
            Line2D([0], [0], color="#333333", lw=1.5, ls="--", label="debiased"),
        ]
        ax.legend(handles=style_handles, fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _plot_combined_decay(
    curves: pd.DataFrame,
    out_pdf: Path,
    title: str,
    x_max: float,
    y_limits: Tuple[float, float] | None,
    show_ylabel: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    methods = [
        ("flow_sinkhorn_sparse", "-", 1.9, "flow Sinkhorn"),
        ("vanilla_sinkhorn_dense", "--", 1.8, "vanilla Sinkhorn"),
    ]
    for method, ls, lw, _ in methods:
        sub = curves[curves["method"] == method].copy()
        eps_vals = sorted(sub["gamma"].dropna().unique())
        for i, eps in enumerate(eps_vals):
            ss = sub[sub["gamma"] == eps].sort_values("runtime_sec")
            x = ss["runtime_sec"].to_numpy(dtype=np.float64)
            y = ss["dual_l2_rel_vs_lp"].to_numpy(dtype=np.float64)
            good = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y > 0)
            if np.sum(good) < 2:
                continue
            ax.plot(
                x[good],
                _best_so_far(y[good]),
                lw=lw,
                ls=ls,
                color=_eps_color(i, len(eps_vals)),
            )

    # Keep panel clean for paper integration.
    if title:
        ax.set_title(title)
    ax.set_xlabel("wall clock time (sec)")
    if show_ylabel:
        ax.set_ylabel("best-so-far relative centered dual L2 error")
    else:
        ax.set_ylabel("")
    ax.set_yscale("log")
    if np.isfinite(x_max) and x_max > 0:
        ax.set_xlim(0.0, x_max)
    if y_limits is not None and all(np.isfinite(y_limits)) and y_limits[0] > 0 and y_limits[1] > y_limits[0]:
        ax.set_ylim(*y_limits)
    ax.grid(alpha=0.25)
    method_legend = [
        Line2D([0], [0], color="#333333", lw=1.9, ls="-", label="flow Sinkhorn"),
        Line2D([0], [0], color="#333333", lw=1.8, ls="--", label="vanilla Sinkhorn"),
    ]
    leg0 = ax.legend(handles=method_legend, fontsize=8, frameon=False, loc="upper right")
    ax.add_artist(leg0)
    # No gamma legend in the paper panels.
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _common_error_ylim(curves: pd.DataFrame) -> Tuple[float, float] | None:
    cols = ["dual_l2_rel_vs_lp"]
    if "dual_l2_debiased_rel_vs_lp" in curves.columns:
        cols.append("dual_l2_debiased_rel_vs_lp")
    vals = []
    for col in cols:
        y = curves[col].to_numpy(dtype=np.float64)
        vals.append(y[np.isfinite(y) & (y > 0)])
    vals = [v for v in vals if v.size > 0]
    if not vals:
        return None
    all_vals = np.concatenate(vals)
    ymin = float(np.nanmin(all_vals))
    ymax = float(np.nanmax(all_vals))
    return max(ymin * 0.75, 1e-8), ymax * 1.35


def _plot_flow_on_graph(problem: Dict[str, object], flow_viz: List[Dict[str, object]], out_pdf: Path) -> None:
    pts = np.asarray(problem["points"], dtype=np.float64)
    gi = np.asarray(problem["graph_indices"], dtype=np.int64)
    mu0 = np.asarray(problem["mus"][0], dtype=np.float64)
    mu1 = np.asarray(problem["mus"][1], dtype=np.float64)
    src = np.where(mu0 > 0)[0]
    tgt = np.where(mu1 > 0)[0]

    ncols = min(2, max(1, len(flow_viz)))
    nrows = int(np.ceil(len(flow_viz) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 5.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, item in zip(axes_flat, flow_viz):
        eps = float(item["gamma"])
        eidx = np.asarray(item["edge_indices"], dtype=np.int64)
        fv = np.asarray(item["flow_values"], dtype=np.float64)

        und = _net_undirected_flow(eidx, fv)

        vals = np.asarray(list(und.values()), dtype=np.float64)
        thr = float(np.quantile(vals, 0.84)) if vals.size > 0 else np.inf
        vmax = float(np.max(vals)) if vals.size > 0 else 1.0
        vmax = max(vmax, 1e-12)

        for i, j in zip(gi[0], gi[1]):
            if i < j:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#D5D9DE", lw=0.45, zorder=1)

        for (u, v), mag in und.items():
            if mag < thr:
                continue
            width = 0.6 + 3.2 * (mag / vmax)
            ax.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]], color="#E67E22", lw=width, alpha=0.95, zorder=2)

        ax.scatter(pts[:, 0], pts[:, 1], s=8, color="#9CA6B0", alpha=0.55, zorder=0)
        ax.scatter(pts[src, 0], pts[src, 1], s=25, color="#1f77b4", zorder=3)
        ax.scatter(pts[tgt, 0], pts[tgt, 1], s=25, color="#d62728", zorder=3)
        ax.set_title(rf"Flow-Sinkhorn flow ($\gamma={eps:.3g}$)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.2)

    for ax in axes_flat[len(flow_viz):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _plot_medium_flow(problem: Dict[str, object], flow_item: Dict[str, object] | None, out_pdf: Path) -> None:
    pts = np.asarray(problem["points"], dtype=np.float64)
    gi = np.asarray(problem["graph_indices"], dtype=np.int64)
    mu0 = np.asarray(problem["mus"][0], dtype=np.float64)
    mu1 = np.asarray(problem["mus"][1], dtype=np.float64)
    src = np.where(mu0 > 0)[0]
    tgt = np.where(mu1 > 0)[0]
    if flow_item is None:
        fig, _ = plt.subplots(figsize=(6.8, 5.0))
        fig.savefig(out_pdf)
        fig.savefig(out_pdf.with_suffix(".png"), dpi=220)
        plt.close(fig)
        return

    item = flow_item
    eidx = np.asarray(item["edge_indices"], dtype=np.int64)
    fv = np.asarray(item["flow_values"], dtype=np.float64)
    und = _net_undirected_flow(eidx, fv)
    vals = np.asarray(list(und.values()), dtype=np.float64)
    thr = float(np.quantile(vals, 0.84)) if vals.size > 0 else np.inf
    vmax = float(np.max(vals)) if vals.size > 0 else 1.0
    vmax = max(vmax, 1e-12)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    for i, j in zip(gi[0], gi[1]):
        if i < j:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color="#D5D9DE", lw=0.42, zorder=1)
    for (u, v), mag in und.items():
        if mag < thr:
            continue
        width = 0.65 + 3.1 * (mag / vmax)
        ax.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]], color="#E67E22", lw=width, alpha=0.95, zorder=2)
    ax.scatter(pts[:, 0], pts[:, 1], s=9, color="#9CA6B0", alpha=0.5, zorder=0)
    ax.scatter(pts[src, 0], pts[src, 1], s=42, color="#1f77b4", zorder=3)
    ax.scatter(pts[tgt, 0], pts[tgt, 1], s=42, color="#d62728", zorder=3)
    # Keep panel clean for paper integration.
    ax.grid(alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _plot_epsilon_summary(summary: pd.DataFrame, out_pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    markers = {"flow_sinkhorn_sparse": "o", "vanilla_sinkhorn_dense": "s"}
    colors = {"flow_sinkhorn_sparse": "#1f77b4", "vanilla_sinkhorn_dense": "#ff7f0e"}
    for method, ss in summary.groupby("method"):
        ss = ss.sort_values("gamma")
        ax.plot(
            ss["gamma"],
            ss["best_final_dual_l2_rel"],
            marker=markers.get(method, "o"),
            lw=1.8,
            color=colors.get(method, "#333333"),
            label=method,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("final relative centered dual L2 error")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _write_report_tex(
    bench: str,
    meta: Dict[str, object],
    out_tex: Path,
    figure_problem: Path,
    figure_iter: Path,
    figure_flow: Path,
    figure_eps: Path,
    table_tex: Path,
) -> None:
    def _latex_escape(s: str) -> str:
        repl = {
            "\\": r"\textbackslash{}",
            "{": r"\{",
            "}": r"\}",
            "_": r"\_",
            "%": r"\%",
            "&": r"\&",
            "#": r"\#",
            "$": r"\$",
            "^": r"\^{}",
            "~": r"\~{}",
        }
        out = s
        for k, v in repl.items():
            out = out.replace(k, v)
        return out

    bench_tex = _latex_escape(str(bench))
    meta_tex = _latex_escape(str(meta))
    tex = f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{hyperref}}

\\title{{Benchmark Report: {bench_tex}}}
\\author{{Flow Sinkhorn benchmark runner}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section{{Setup}}
Benchmark type: \\texttt{{{bench_tex}}}.\\\\
Metadata: \\texttt{{{meta_tex}}}.

\\section{{Graph and Measures}}
\\begin{{figure}}[h!]
  \\centering
  \\includegraphics[width=0.9\\textwidth]{{{figure_problem.as_posix()}}}
  \\caption{{Graph geometry and endpoint measure supports.}}
\\end{{figure}}

\\section{{Iteration-Level Curves}}
\\begin{{figure}}[h!]
  \\centering
  \\includegraphics[width=0.98\\textwidth]{{{figure_iter.as_posix()}}}
  \\caption{{Best-so-far relative centered dual $L_2$ error vs runtime, one curve per $\\gamma$.}}
\\end{{figure}}

\\section{{Flow on Graph for Several Epsilons}}
\\begin{{figure}}[h!]
  \\centering
  \\includegraphics[width=0.98\\textwidth]{{{figure_flow.as_posix()}}}
  \\caption{{Flow-Sinkhorn transport flow rendered on the graph for converged final iterates at several $\\gamma$ values.}}
\\end{{figure}}

\\section{{Final Error vs Regularization}}
\\begin{{figure}}[h!]
  \\centering
  \\includegraphics[width=0.72\\textwidth]{{{figure_eps.as_posix()}}}
  \\caption{{Final relative centered dual $L_2$ error as a function of $\\gamma$.}}
\\end{{figure}}

\\section{{Summary Table}}
\\input{{{table_tex.as_posix()}}}

\\end{{document}}
"""
    out_tex.write_text(tex)


def _compile_pdf(tex_path: Path) -> None:
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    subprocess.run(cmd, cwd=tex_path.parent, check=True, capture_output=True, text=True)
    subprocess.run(cmd, cwd=tex_path.parent, check=True, capture_output=True, text=True)


def main() -> None:
    args = parse_args()
    if args.require_gpu and not args.use_torch:
        raise RuntimeError("--require-gpu requires --use-torch (GPU-only execution).")
    t0 = time.perf_counter()

    out_root = Path(args.out_root)
    fig_dir_paper = out_root / "figures"
    fig_dir_detail = Path("benchmarks/results/figures")
    tab_dir = Path("benchmarks/results/tables")
    fig_dir_paper.mkdir(parents=True, exist_ok=True)
    fig_dir_detail.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    problem = prepare_problem(args)
    result = run_benchmark(problem, args)

    bench = str(args.bench)
    curves = result["curves"]
    summary = result["summary"]

    curves_csv = tab_dir / f"report-{bench}-curves.csv"
    summary_csv = tab_dir / f"report-{bench}-summary.csv"
    summary_tex = tab_dir / f"report-{bench}-summary.tex"

    curves.to_csv(curves_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_latex = summary.to_latex(index=False, float_format=lambda x: f"{x:.6g}", escape=True)
    summary_tex.write_text(summary_latex)

    fig_problem = fig_dir_detail / f"report-{bench}-problem.pdf"
    fig_iter_flow = fig_dir_detail / f"report-{bench}-iter-flow.pdf"
    fig_iter_van = fig_dir_detail / f"report-{bench}-iter-vanilla.pdf"
    fig_iter_combined = fig_dir_detail / f"report-{bench}-iter-combined.pdf"
    fig_flow = fig_dir_detail / f"report-{bench}-flow-graph.pdf"
    fig_eps = fig_dir_detail / f"report-{bench}-epsilon.pdf"
    fig_row_graph = fig_dir_paper / f"benchmark-{args.setup_tag}-{bench}-graph.pdf"
    fig_row_conv = fig_dir_paper / f"benchmark-{args.setup_tag}-{bench}-convergence.pdf"

    _plot_problem(problem, fig_problem)
    x_max = float(curves["runtime_sec"].max()) if ("runtime_sec" in curves.columns and len(curves) > 0) else np.nan
    y_limits = _common_error_ylim(curves)
    _plot_error_decay(curves, "flow_sinkhorn_sparse", fig_iter_flow, "Flow-Sinkhorn error decay", x_max=x_max, y_limits=y_limits)
    _plot_error_decay(curves, "vanilla_sinkhorn_dense", fig_iter_van, "Vanilla Sinkhorn error decay", x_max=x_max, y_limits=y_limits)
    _plot_combined_decay(curves, fig_iter_combined, "", x_max=x_max, y_limits=y_limits, show_ylabel=True)
    _plot_flow_on_graph(problem, result.get("flow_viz", []), fig_flow)
    _plot_medium_flow(problem, result.get("medium_flow"), fig_row_graph)
    _plot_combined_decay(
        curves,
        fig_row_conv,
        "",
        x_max=x_max,
        y_limits=y_limits,
        show_ylabel=(bench == "line"),
    )
    _plot_epsilon_summary(summary, fig_eps)

    report_root = Path("benchmarks/results")
    report_root.mkdir(parents=True, exist_ok=True)
    report_tex = report_root / f"report-{bench}.tex"
    _write_report_tex(
        bench=bench,
        meta=problem["meta"],
        out_tex=report_tex,
        figure_problem=Path(os.path.relpath(fig_problem, report_root)),
        figure_iter=Path(os.path.relpath(fig_iter_flow, report_root)),
        figure_flow=Path(os.path.relpath(fig_flow, report_root)),
        figure_eps=Path(os.path.relpath(fig_eps, report_root)),
        table_tex=Path(os.path.relpath(summary_tex, report_root)),
    )
    try:
        _compile_pdf(report_tex)
    except FileNotFoundError:
        print("[warn] pdflatex not found, skipping LaTeX report compilation.")

    dt = time.perf_counter() - t0
    print(f"[ok] bench={bench}")
    print(f"[ok] curves: {curves_csv}")
    print(f"[ok] summary: {summary_csv}")
    print(f"[ok] pdf: {report_root / f'report-{bench}.pdf'}")
    print(f"[ok] elapsed_sec={dt:.2f}")


if __name__ == "__main__":
    main()
