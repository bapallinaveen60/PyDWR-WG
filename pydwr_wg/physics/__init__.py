"""
Radar physics, scattering theory, self-consistency, QPE estimators, and microphysics.
"""

from pydwr_wg.physics.self_consistency import PolarimetricSelfConsistency
from pydwr_wg.physics.qpe_estimators import (
    r_marshall_palmer,
    r_western_ghats_warm,
    r_kdp_sband,
    r_zh_zdr_sband,
    r_kdp_zdr_sband,
    adaptive_polarimetric_qpe,
)
from pydwr_wg.physics.microphysics import WarmRainMicrophysics

__all__ = [
    "PolarimetricSelfConsistency",
    "r_marshall_palmer",
    "r_western_ghats_warm",
    "r_kdp_sband",
    "r_zh_zdr_sband",
    "r_kdp_zdr_sband",
    "adaptive_polarimetric_qpe",
    "WarmRainMicrophysics",
]
