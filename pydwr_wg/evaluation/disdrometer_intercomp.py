"""
In-situ Microphysical Validation and Escarpment Benchmarking at HACPL Mahabaleshwar.
Intercompares radar reconstructed boundary-layer profiles against JWD, 2DVD, and MRR-2 vertical profiles.
"""

import numpy as np
from typing import Dict, Any, Tuple
from pydwr_wg.constants import HACPL_MAHABALESHWAR

class DisdrometerIntercomparison:
    """
    Benchmarks radar de-cluttered and reconstructed profiles against HACPL surface disdrometer
    and Micro Rain Radar (MRR-2) vertical profiles.
    """

    def __init__(
        self,
        radar_lat: float,
        radar_lon: float,
        radar_alt_m: float = 60.0,
    ):
        self.radar_lat = radar_lat
        self.radar_lon = radar_lon
        self.radar_alt_m = radar_alt_m
        self.hacpl_lat = HACPL_MAHABALESHWAR["latitude"]
        self.hacpl_lon = HACPL_MAHABALESHWAR["longitude"]
        self.hacpl_alt_m = HACPL_MAHABALESHWAR["altitude_m"]

        # Calculate range and azimuth from radar to HACPL
        self.distance_km, self.azimuth_deg = self._calc_geodesic_distance(
            self.radar_lat, self.radar_lon, self.hacpl_lat, self.hacpl_lon
        )

    def evaluate_boundary_layer_bias(
        self,
        radar_reconstructed_zh: float,
        mrr_heights_agl: np.ndarray,
        mrr_zh_profile: np.ndarray,
        max_agl_height_m: float = 1500.0,
    ) -> Dict[str, float]:
        """
        Calculates reflectivity bias in the lowest 1.5 km boundary layer:
        Bias = ZH_radar - mean(ZH_mrr[h < 1500m])

        Target threshold: [-1.0, +1.0] dB (vs operational baseline -6.5 to -8.5 dB from beam overshooting).
        """
        bl_mask = (mrr_heights_agl <= max_agl_height_m) & (~np.isnan(mrr_zh_profile))
        if np.sum(bl_mask) == 0:
            return {"boundary_layer_bias_db": 0.0, "mrr_bl_mean_zh": 0.0}

        mrr_bl_mean = float(np.mean(mrr_zh_profile[bl_mask]))
        bias_db = float(radar_reconstructed_zh - mrr_bl_mean)

        return {
            "boundary_layer_bias_db": bias_db,
            "mrr_bl_mean_zh": mrr_bl_mean,
            "radar_reconstructed_zh": float(radar_reconstructed_zh),
            "meets_target_spec": abs(bias_db) <= 1.0,
        }

    def evaluate_surface_rain_rate(
        self,
        radar_rain_rate: float,
        disdrometer_rain_rate: float,
    ) -> Dict[str, float]:
        """
        Evaluates surface precipitation estimation against surface disdrometer.
        """
        abs_err = abs(radar_rain_rate - disdrometer_rain_rate)
        rel_err = (abs_err / disdrometer_rain_rate * 100.0) if disdrometer_rain_rate > 0.1 else 0.0

        return {
            "radar_r_mm_h": float(radar_rain_rate),
            "gauge_r_mm_h": float(disdrometer_rain_rate),
            "abs_error_mm_h": float(abs_err),
            "rel_error_percent": float(rel_err),
        }

    @staticmethod
    def _calc_geodesic_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> Tuple[float, float]:
        # Approximate distance and azimuth
        km_lat = 111.132 * (lat2 - lat1)
        km_lon = 111.320 * np.cos(np.radians((lat1 + lat2) / 2.0)) * (lon2 - lon1)
        dist_km = float(np.sqrt(km_lat**2 + km_lon**2))
        az_deg = float(np.degrees(np.arctan2(km_lon, km_lat)) % 360.0)
        return dist_km, az_deg
