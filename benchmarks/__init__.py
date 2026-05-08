"""Utilities for single-cell Waddington-OT benchmarking with Flow Sinkhorn."""

from .wot_benchmark import (
    BenchmarkConfig,
    benchmark_methods_on_prepared_data,
    propose_epsilon_candidates,
    run_flow_sinkhorn_sparse_trajectory,
    run_vanilla_sinkhorn_dense_trajectory,
    screen_stable_epsilons,
    download_wot_tutorial_data,
    ensure_wot_data_available,
    floyd_warshall_metric,
    interpolate_from_endpoint_coupling,
    prepare_wot_data,
    run_benchmark,
    solve_graph_w1_lp,
    summarize_benchmark,
)

__all__ = [
    "BenchmarkConfig",
    "benchmark_methods_on_prepared_data",
    "propose_epsilon_candidates",
    "run_flow_sinkhorn_sparse_trajectory",
    "run_vanilla_sinkhorn_dense_trajectory",
    "screen_stable_epsilons",
    "download_wot_tutorial_data",
    "ensure_wot_data_available",
    "floyd_warshall_metric",
    "interpolate_from_endpoint_coupling",
    "prepare_wot_data",
    "run_benchmark",
    "solve_graph_w1_lp",
    "summarize_benchmark",
]
