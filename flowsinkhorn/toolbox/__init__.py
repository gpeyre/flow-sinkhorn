"""
Reusable helper utilities for examples and experiments.

The toolbox centralizes graph-construction, source/sink generation, and
visualization helpers that were previously embedded in notebooks.
"""

from .grid import (
    create_grid_graph,
    gaussian_bump,
    compute_modulated_costs,
    create_corner_sources_sinks,
    plot_grid_with_costs,
    plot_grid_with_flow,
)
from .planar import (
    create_planar_knn_graph,
    create_source_sink_from_positions,
)
from .mesh import (
    load_off_file,
    build_mesh_graph,
    select_sources_sinks,
    plot_mesh,
    plot_mesh_with_flow,
)

__all__ = [
    "create_grid_graph",
    "gaussian_bump",
    "compute_modulated_costs",
    "create_corner_sources_sinks",
    "plot_grid_with_costs",
    "plot_grid_with_flow",
    "create_planar_knn_graph",
    "create_source_sink_from_positions",
    "load_off_file",
    "build_mesh_graph",
    "select_sources_sinks",
    "plot_mesh",
    "plot_mesh_with_flow",
]
