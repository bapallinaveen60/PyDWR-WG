"""
Preprocessing routines: 3D ray-tracing, phase unwrapping, Kalman KDP, clutter compositing, coordinate transform.
"""

from pydwr_wg.preprocessing.ray_tracing import (
    compute_beam_height,
    compute_ground_range,
    compute_partial_beam_blockage,
)
from pydwr_wg.preprocessing.phase_processing import (
    unwrap_phidp,
    estimate_kdp_kalman,
    filter_backscatter_phase_shift,
)
from pydwr_wg.preprocessing.clutter_climatology import (
    build_clutter_climatology,
    generate_main_bang_mask,
)
from pydwr_wg.preprocessing.coordinate_transform import (
    polar_to_cartesian_coords,
    polar_to_cartesian_grid,
)

__all__ = [
    "compute_beam_height",
    "compute_ground_range",
    "compute_partial_beam_blockage",
    "unwrap_phidp",
    "estimate_kdp_kalman",
    "filter_backscatter_phase_shift",
    "build_clutter_climatology",
    "generate_main_bang_mask",
    "polar_to_cartesian_coords",
    "polar_to_cartesian_grid",
]
