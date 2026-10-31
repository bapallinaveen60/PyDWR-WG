"""
PyDWR-WG: Ground Polarimetric Doppler Radar De-Cluttering and Low-Level Warm Orographic
Precipitation Reconstruction Over the Western Ghats.

An operational framework for high-fidelity satellite ground-truthing under complex terrain.
"""

__version__ = "0.1.0"
__author__ = "GeoVision Research Group"

from pydwr_wg.constants import RADAR_SITES, EARTH_RADIUS_4_3, DIELECTRIC_FACTOR_KW2
from pydwr_wg.models.pta_innet import PTAInNet
from pydwr_wg.models.losses import TotalPTALoss

__all__ = [
    "__version__",
    "RADAR_SITES",
    "EARTH_RADIUS_4_3",
    "DIELECTRIC_FACTOR_KW2",
    "PTAInNet",
    "TotalPTALoss",
]
