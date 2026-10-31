"""
INSAT-3DR/3DS Imager Ingestion and Parallax Correction Engine.
Handles TIR1, TIR2, MIR, WV channels and applies geometry-based cloud-top parallax shifts.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple

class INSATLoader:
    """
    Ingests INSAT-3DR / INSAT-3DS Imager multispectral observations and applies
    orographic cloud parallax shift corrections.
    """

    def __init__(
        self,
        sub_satellite_lon: float = 74.0,  # INSAT-3DR geostationary sub-satellite longitude
        sub_satellite_lat: float = 0.0,
        satellite_altitude_km: float = 35786.0,
    ):
        self.sub_sat_lon = sub_satellite_lon
        self.sub_sat_lat = sub_satellite_lat
        self.sat_alt_km = satellite_altitude_km

    def compute_parallax_correction(
        self,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        cloud_top_height_km: float = 3.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes geostationary viewing parallax correction displacement for shallow orographic clouds.

        Parameters
        ----------
        target_lat : np.ndarray
            Observed cloud pixel latitudes.
        target_lon : np.ndarray
            Observed cloud pixel longitudes.
        cloud_top_height_km : float
            Assumed or measured cloud-top height above ground level (e.g. 3.5 km for Western Ghats warm rain).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Parallax-corrected (true_lat, true_lon).
        """
        delta_lon_rad = np.radians(target_lon - self.sub_sat_lon)
        lat_rad = np.radians(target_lat)

        # Geocentric angles
        cos_beta = np.cos(lat_rad) * np.cos(delta_lon_rad)
        sin_beta = np.sqrt(1.0 - np.clip(cos_beta**2, 0.0, 1.0))

        # Satellite view angle theta
        re_km = 6371.0
        distance_to_sat = np.sqrt(
            re_km**2 + (re_km + self.sat_alt_km)**2 - 2.0 * re_km * (re_km + self.sat_alt_km) * cos_beta
        )
        sin_theta = (re_km + self.sat_alt_km) * sin_beta / distance_to_sat
        tan_theta = sin_theta / np.sqrt(np.maximum(1.0 - sin_theta**2, 1e-6))

        # Ground displacement
        displacement_km = cloud_top_height_km * tan_theta

        # Direction vector away from sub-satellite point
        phi = np.arctan2(np.sin(delta_lon_rad), np.tan(lat_rad))

        d_lat_km = -displacement_km * np.cos(phi)
        d_lon_km = -displacement_km * np.sin(phi)

        km_per_deg_lat = 111.132
        km_per_deg_lon = 111.320 * np.cos(lat_rad)

        true_lat = target_lat + (d_lat_km / km_per_deg_lat)
        true_lon = target_lon + (d_lon_km / km_per_deg_lon)

        return true_lat, true_lon

    def generate_synthetic_scene(
        self,
        n_lat: int = 200,
        n_lon: int = 200,
        warm_rain_active: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generates synthetic 4-channel INSAT-3DR Imager observations over the Western Ghats.
        Demonstrates the warm-cloud ambiguity: precipitating warm clouds have TIR1 between 265 and 282 K.
        """
        # TIR1 (10.8 um): 265 - 285 K for shallow warm clouds; 295 - 305 K for clear land/sea
        if warm_rain_active:
            tir1 = 274.0 + np.random.normal(0, 3.0, (n_lat, n_lon))
            tir2 = tir1 - 1.2 + np.random.normal(0, 0.3, (n_lat, n_lon))
            mir = 285.0 + np.random.normal(0, 4.0, (n_lat, n_lon))
            wv = 245.0 + np.random.normal(0, 2.5, (n_lat, n_lon))
        else:
            tir1 = 298.0 + np.random.normal(0, 2.0, (n_lat, n_lon))
            tir2 = tir1 - 0.8 + np.random.normal(0, 0.2, (n_lat, n_lon))
            mir = 305.0 + np.random.normal(0, 3.0, (n_lat, n_lon))
            wv = 238.0 + np.random.normal(0, 3.0, (n_lat, n_lon))

        return {
            "TIR1": tir1.astype(np.float32),
            "TIR2": tir2.astype(np.float32),
            "MIR": mir.astype(np.float32),
            "WV": wv.astype(np.float32),
        }
