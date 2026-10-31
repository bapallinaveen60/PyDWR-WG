"""
Evaluation metrics, baseline benchmarks, and HACPL disdrometer intercomparison.
"""

from pydwr_wg.evaluation.metrics import (
    compute_categorical_metrics,
    compute_low_velocity_rain_preservation,
    compute_reconstruction_fidelity,
    compute_qpe_metrics,
)
from pydwr_wg.evaluation.benchmarks import (
    gmap_notch_filter,
    gabella_texture_filter,
    fuzzy_logic_cmd,
)
from pydwr_wg.evaluation.disdrometer_intercomp import DisdrometerIntercomparison

__all__ = [
    "compute_categorical_metrics",
    "compute_low_velocity_rain_preservation",
    "compute_reconstruction_fidelity",
    "compute_qpe_metrics",
    "gmap_notch_filter",
    "gabella_texture_filter",
    "fuzzy_logic_cmd",
    "DisdrometerIntercomparison",
]
