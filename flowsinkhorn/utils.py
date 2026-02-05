"""
Backward-compatible imports for helper utilities.

This module is kept for compatibility. New code should import helpers from:
    flowsinkhorn.toolbox
or:
    flowsinkhorn.toolbox.mesh
"""

from .toolbox.mesh import (
    load_off_file,
    build_mesh_graph,
    select_sources_sinks,
    plot_mesh,
    plot_mesh_with_flow,
)

__all__ = [
    "load_off_file",
    "build_mesh_graph",
    "select_sources_sinks",
    "plot_mesh",
    "plot_mesh_with_flow",
]
