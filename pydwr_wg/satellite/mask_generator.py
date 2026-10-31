"""
5-Class Precipitation Reference Mask Synthesis for Satellite Ground-Truthing.
Generates objective meteorological labels from quality-controlled ground radar composites.
"""

import numpy as np
from scipy.ndimage import label
from typing import Optional
from pydwr_wg.constants import PRECIPITATION_CLASSES

class PrecipitationMaskGenerator:
    """
    Synthesizes a 5-class precipitation mask from polarimetric radar reflectivity,
    echo tops, and spatial morphology.
    """

    def __init__(
        self,
        rain_threshold_dbz: float = 15.0,
        deep_convective_dbz: float = 40.0,
        shallow_max_top_km: float = 4.5,
        isolated_cell_area_max_km2: float = 64.0,  # <= 8x8 km
    ):
        self.rain_thresh = rain_threshold_dbz
        self.deep_thresh = deep_convective_dbz
        self.shallow_max_top = shallow_max_top_km
        self.isolated_area_max = isolated_cell_area_max_km2

    def generate_5class_mask(
        self,
        zh_1km: np.ndarray,
        echo_top_height_km: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Synthesizes the categorical reference mask [Ny, Nx]:
        0: No-Rain
        1: Stratiform
        2: Deep Convective
        3: Shallow Isolated Convective
        4: Shallow Non-Isolated Convective

        Parameters
        ----------
        zh_1km : np.ndarray
            Reconstructed radar reflectivity on 1 km Cartesian grid [Ny, Nx].
        echo_top_height_km : Optional[np.ndarray]
            Echo top height in km MSL. If None, estimated from reflectivity morphology.

        Returns
        -------
        np.ndarray
            5-class categorical mask [Ny, Nx] (dtype: uint8).
        """
        ny, nx = zh_1km.shape
        mask = np.zeros((ny, nx), dtype=np.uint8)

        is_rain = (zh_1km >= self.rain_thresh) & (~np.isnan(zh_1km))
        if np.sum(is_rain) == 0:
            return mask

        if echo_top_height_km is None:
            # Empirical echo top proxy: warm orographic clouds over Western Ghats are mostly < 4.5 km
            # Deep convective towers exceed 8.0 km when ZH > 40 dBZ
            echo_top_height_km = np.where(
                zh_1km >= self.deep_thresh, 10.0,
                np.where(zh_1km >= 25.0, 3.8, 3.0)
            )

        # Class 2: Deep Convective
        is_deep = is_rain & (zh_1km >= self.deep_thresh) & (echo_top_height_km >= 7.0)
        mask[is_deep] = 2

        # Candidate shallow warm convective: echo top < 4.5 km, reflectivity 18-38 dBZ
        is_shallow = is_rain & (echo_top_height_km <= self.shallow_max_top) & (~is_deep)

        # Distinguish Shallow Isolated (3) vs Shallow Non-Isolated (4) via connected components
        labeled_shallow, num_features = label(is_shallow)
        for feat_id in range(1, num_features + 1):
            cell_mask = labeled_shallow == feat_id
            cell_area_km2 = float(np.sum(cell_mask))  # each pixel = 1 km^2
            if cell_area_km2 <= self.isolated_area_max:
                mask[cell_mask] = 3  # Shallow Isolated Convective
            else:
                mask[cell_mask] = 4  # Shallow Non-Isolated Convective (continuous orographic rainband)

        # Class 1: Stratiform (remaining authentic precipitation)
        is_stratiform = is_rain & (mask == 0)
        mask[is_stratiform] = 1

        return mask
