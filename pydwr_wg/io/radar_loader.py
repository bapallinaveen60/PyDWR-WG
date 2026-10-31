"""
Radar Volume Ingestion and Polar Array Standardization.
Supports IMD Rainbow, IRIS, NetCDF/CF-Radial via Py-ART with robust fallback mechanisms.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pydwr_wg.constants import (
    DEFAULT_N_RANGE_GATES,
    DEFAULT_RANGE_STEP_M,
    DEFAULT_N_AZIMUTHS,
    DEFAULT_AZIMUTH_STEP_DEG,
)

class RadarDataLoader:
    """
    Standardizes raw operational radar volumes into unified polar tensors [Nr, Na].
    """

    def __init__(
        self,
        n_range_gates: int = DEFAULT_N_RANGE_GATES,
        range_step_m: float = DEFAULT_RANGE_STEP_M,
        n_azimuths: int = DEFAULT_N_AZIMUTHS,
        azimuth_step_deg: float = DEFAULT_AZIMUTH_STEP_DEG,
    ):
        self.n_range_gates = n_range_gates
        self.range_step_m = range_step_m
        self.n_azimuths = n_azimuths
        self.azimuth_step_deg = azimuth_step_deg
        self.target_ranges_km = np.arange(1, n_range_gates + 1) * (range_step_m / 1000.0)
        self.target_azimuths_deg = np.arange(0, n_azimuths) * azimuth_step_deg

    def load_sweep(
        self,
        filepath: str,
        target_elevation_deg: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Loads and extracts a single elevation tilt from a radar file, resampling onto
        the standardized polar grid [Nr, Na].

        Parameters
        ----------
        filepath : str
            Path to radar file (NetCDF, HDF5, Rainbow, or IRIS).
        target_elevation_deg : float
            Elevation angle in degrees to extract.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 2D arrays [Nr, Na]:
            - 'ZH': Horizontal reflectivity [dBZ]
            - 'ZDR': Differential reflectivity [dB]
            - 'RHOHV': Copolar correlation coefficient
            - 'PHIDP': Differential propagation phase [deg]
            - 'VR': Radial velocity [m/s]
            - 'SW': Spectrum width [m/s]
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Radar file not found: {filepath}")

        # Try pyart if available
        try:
            import pyart
            radar = pyart.io.read(filepath)
            return self._extract_pyart_sweep(radar, target_elevation_deg)
        except (ImportError, Exception):
            # Fallback for NetCDF / HDF5 using netCDF4 / h5py
            return self._extract_netcdf_sweep(filepath, target_elevation_deg)

    def _extract_pyart_sweep(self, radar, target_elevation_deg: float) -> Dict[str, np.ndarray]:
        """Extracts and resamples sweep from Py-ART Radar object."""
        sweep_idx = np.argmin(np.abs(radar.fixed_angle['data'] - target_elevation_deg))
        sweep_slice = radar.get_slice(sweep_idx)

        # Get radar fields
        field_keys = {
            "ZH": ["reflectivity", "total_power", "DBZ"],
            "ZDR": ["differential_reflectivity", "ZDR"],
            "RHOHV": ["cross_correlation_ratio", "RHOHV"],
            "PHIDP": ["differential_phase", "PHIDP"],
            "VR": ["velocity", "VEL"],
            "SW": ["spectrum_width", "WIDTH"],
        }

        output = {}
        raw_az = radar.azimuth['data'][sweep_slice]
        raw_rng = radar.range['data'] / 1000.0  # km

        for key, aliases in field_keys.items():
            field_data = None
            for alias in aliases:
                if alias in radar.fields:
                    field_data = radar.fields[alias]['data'][sweep_slice]
                    break

            if field_data is None:
                output[key] = np.zeros((self.n_range_gates, self.n_azimuths), dtype=np.float32)
            else:
                output[key] = self._resample_to_grid(raw_rng, raw_az, field_data)

        return output

    def _extract_netcdf_sweep(self, filepath: str, target_elevation_deg: float) -> Dict[str, np.ndarray]:
        """Direct extraction fallback using netCDF4 / h5py."""
        import netCDF4 as nc

        with nc.Dataset(filepath, "r") as ds:
            # Check variables
            output = {}
            for name in ["ZH", "ZDR", "RHOHV", "PHIDP", "VR", "SW"]:
                if name in ds.variables:
                    data = ds.variables[name][:]
                    # Ensure shape [Nr, Na]
                    if data.ndim == 2:
                        if data.shape == (self.n_range_gates, self.n_azimuths):
                            output[name] = np.array(data, dtype=np.float32)
                        elif data.shape == (self.n_azimuths, self.n_range_gates):
                            output[name] = np.array(data.T, dtype=np.float32)
                        else:
                            output[name] = self._resize_array(data)
                    else:
                        output[name] = np.zeros((self.n_range_gates, self.n_azimuths), dtype=np.float32)
                else:
                    output[name] = np.zeros((self.n_range_gates, self.n_azimuths), dtype=np.float32)

        return output

    def _resample_to_grid(
        self,
        raw_range_km: np.ndarray,
        raw_azimuth_deg: np.ndarray,
        raw_data: np.ndarray,
    ) -> np.ndarray:
        """Resamples raw radar sweep data onto target [Nr, Na] polar grid."""
        # Simple nearest-neighbor lookup for fast operational performance
        out = np.zeros((self.n_range_gates, self.n_azimuths), dtype=np.float32)
        r_indices = np.clip(
            np.searchsorted(raw_range_km, self.target_ranges_km), 0, len(raw_range_km) - 1
        )

        for a_idx, target_az in enumerate(self.target_azimuths_deg):
            az_diff = np.abs((raw_azimuth_deg - target_az + 180.0) % 360.0 - 180.0)
            closest_ray = np.argmin(az_diff)
            out[:, a_idx] = raw_data[closest_ray, r_indices]

        return out

    def _resize_array(self, data: np.ndarray) -> np.ndarray:
        from scipy.ndimage import zoom
        zoom_factors = (
            self.n_range_gates / data.shape[0],
            self.n_azimuths / data.shape[1],
        )
        return zoom(data, zoom_factors, order=1).astype(np.float32)

    def generate_synthetic_marine_sweep(self) -> Dict[str, np.ndarray]:
        """
        Generates a synthetic pristine marine warm-rain sweep over the Arabian Sea
        for offline testing and SCIP development.
        """
        nr, na = self.n_range_gates, self.n_azimuths
        r = np.linspace(1, 100, nr)[:, np.newaxis]
        az = np.linspace(0, 360, na)[np.newaxis, :]

        # Offshore rainband located in azimuth sector 200 - 320 deg
        band_mask = np.sin(np.radians(az * 2.0)) > 0.2
        r_center = 45.0
        r_profile = np.exp(-((r - r_center) ** 2) / (2.0 * (15.0**2)))

        # Reflectivity: 20 to 38 dBZ (warm shallow rain)
        zh = 15.0 + 22.0 * (r_profile * band_mask) + np.random.normal(0, 0.5, (nr, na))
        zh = np.clip(zh, -10.0, 50.0).astype(np.float32)

        # ZDR: 0.2 to 1.4 dB
        zdr = 0.2 + 0.03 * (zh - 15.0) + np.random.normal(0, 0.05, (nr, na))
        zdr = np.clip(zdr, 0.1, 2.0).astype(np.float32)

        # Copolar correlation: 0.97 to 0.99
        rhohv = 0.98 - 0.01 * np.random.uniform(0, 1, (nr, na))
        rhohv = np.clip(rhohv, 0.95, 0.995).astype(np.float32)

        # PhiDP: monotonically increasing across rainband
        kdp = np.maximum(0.01 * (zh - 15.0), 0.0)
        phidp = np.cumsum(2.0 * kdp * (self.range_step_m / 1000.0), axis=0).astype(np.float32)

        # Vr: low onshore radial velocity (-3 to +3 m/s)
        vr = 2.5 * np.cos(np.radians(az - 240.0)) + np.random.normal(0, 0.3, (nr, na))
        vr = vr.astype(np.float32)

        # SW: 1.0 to 2.0 m/s
        sw = 1.4 + np.random.normal(0, 0.2, (nr, na))
        sw = np.clip(sw, 0.5, 3.0).astype(np.float32)

        return {
            "ZH": zh,
            "ZDR": zdr,
            "RHOHV": rhohv,
            "PHIDP": phidp,
            "KDP": kdp.astype(np.float32),
            "VR": vr,
            "SW": sw,
        }
