"""
Microphysical Regime Characterization and Drop Size Distribution Formulations.
Discriminates warm orographic collision-coalescence from deep convective mixed-phase processes.
"""

import numpy as np
from typing import Dict, Any

class WarmRainMicrophysics:
    """
    Formulations for tropical warm orographic drop growth over the Western Ghats.
    """

    @staticmethod
    def estimate_dm_from_polarimetry(zdr_db: np.ndarray) -> np.ndarray:
        """
        Estimates mass-weighted mean diameter Dm [mm] from S-band differential reflectivity ZDR.
        Empirical Western Ghats relation: Dm = 0.98 + 0.52 * ZDR - 0.04 * ZDR^2
        """
        zdr_clip = np.clip(zdr_db, 0.0, 4.0)
        dm = 0.98 + 0.52 * zdr_clip - 0.04 * (zdr_clip ** 2)
        return dm.astype(np.float32)

    @staticmethod
    def estimate_log10_nw(zh_dbz: np.ndarray, dm_mm: np.ndarray) -> np.ndarray:
        """
        Estimates generalized intercept parameter log10(Nw) [mm-1 m-3].
        log10(Nw) = 0.1 * ZH - 3.59 * log10(Dm) - 0.4
        """
        dm_safe = np.maximum(dm_mm, 0.2)
        log10_nw = 0.1 * zh_dbz - 3.59 * np.log10(dm_safe) - 0.4
        return log10_nw.astype(np.float32)

    @staticmethod
    def classify_microphysical_regime(
        zh_dbz: np.ndarray,
        zdr_db: np.ndarray,
        dm_mm: np.ndarray,
        echo_top_height_km: np.ndarray,
        freezing_level_km: float = 5.0,
    ) -> np.ndarray:
        """
        Classifies radar volume gates into meteorological microphysical regimes:
        0: No Rain / Clear Air
        1: Shallow Warm Orographic Rain (Dm ~ 1.0-1.6 mm, echo top < 4.5 km, freezing level > 4.8 km)
        2: Deep Convective Mixed Phase (Echo top > 8 km, high ZH > 40 dBZ, large Dm > 2.0 mm)
        3: Stratiform Monsoonal (Melting layer brightband, moderate ZH ~ 25-35 dBZ)

        Returns
        -------
        np.ndarray
            Integer categorical array [Nr, Na].
        """
        regime = np.zeros_like(zh_dbz, dtype=np.uint8)

        is_rain = zh_dbz > 15.0

        # Shallow warm orographic rain: warm tops below freezing level
        is_warm_shallow = (
            is_rain
            & (echo_top_height_km < freezing_level_km)
            & (zh_dbz < 38.0)
            & (dm_mm <= 1.8)
        )
        regime[is_warm_shallow] = 1

        # Deep convective: strong vertical updrafts, tops well above freezing level
        is_deep_convective = (
            is_rain
            & (echo_top_height_km >= 7.0)
            & (zh_dbz >= 40.0)
        )
        regime[is_deep_convective] = 2

        # Stratiform
        is_stratiform = is_rain & (regime == 0)
        regime[is_stratiform] = 3

        return regime
