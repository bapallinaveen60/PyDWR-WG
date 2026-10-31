"""
Satellite integration routines: INSAT-3DR collocation, 5-class mask synthesis, retraining pipelines.
"""

from pydwr_wg.satellite.collocation import SatelliteCollocationEngine
from pydwr_wg.satellite.mask_generator import PrecipitationMaskGenerator
from pydwr_wg.satellite.retraining_pipeline import SatelliteRetrainingEvaluator

__all__ = [
    "SatelliteCollocationEngine",
    "PrecipitationMaskGenerator",
    "SatelliteRetrainingEvaluator",
]
