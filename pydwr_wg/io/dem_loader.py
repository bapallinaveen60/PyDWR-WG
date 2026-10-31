"""
Digital Elevation Model (DEM) Ingestion and Polar Resampling.
Extracts 30-meter SRTM topography and maps elevations onto radar native polar coordinate systems.
"""

import os
import numpy as np
from typing import Optional, Tuple
from pydwr_wg.preprocessing.coordinate_transform import polar_to_cartesian_coords

class DEMLoader:
    """
    Loads and reprojects high-resolution SRTM DEM topography onto radar polar coordinates [Nr, Na].
    """

    def __init__(
        self,
        radar_lat: float,
        radar_lon: float,
        radar_alt_m: float = 60.0,
        n_range_gates: int = 1000,
        range_step_m: float = 100.0,
        n_azimuths: int = 360,
        azimuth_step_deg: float = 1.0,
    ):
        self.radar_lat = radar_lat
        self.radar_lon = radar_lon
        self.radar_alt_m = radar_alt_m
        self.n_range_gates = n_range_gates
        self.range_step_m = range_step_m
        self.n_azimuths = n_azimuths
        self.azimuth_step_deg = azimuth_step_deg

        self.ranges_km = np.arange(1, n_range_gates + 1) * (range_step_m / 1000.0)
        self.azimuths_deg = np.arange(0, n_azimuths) * azimuth_step_deg

        # Compute geographical coordinates of every radar gate [Nr, Na]
        _, _, self.gate_lats, self.gate_lons = polar_to_cartesian_coords(
            self.ranges_km, self.azimuths_deg, self.radar_lat, self.radar_lon
        )

    def load_dem_polar(self, dem_path: Optional[str] = None) -> np.ndarray:
        """
        Samples DEM onto polar grid. If dem_path is provided and exists, reads via rasterio.
        Otherwise, generates a realistic Western Ghats topographic profile.

        Parameters
        ----------
        dem_path : Optional[str]
            Path to GeoTIFF or NetCDF DEM file.

        Returns
        -------
        np.ndarray
            Elevation array [Nr, Na] in meters MSL.
        """
        if dem_path and os.path.exists(dem_path):
            try:
                import rasterio
                with rasterio.open(dem_path) as src:
                    # Sample points
                    coords = [(lon, lat) for lon, lat in zip(self.gate_lons.ravel(), self.gate_lats.ravel())]
                    elevations = np.array([val[0] for val in src.sample(coords)], dtype=np.float32)
                    dem_polar = elevations.reshape(self.gate_lats.shape)
                    return np.maximum(dem_polar, 0.0)
            except Exception as e:
                print(f"Warning: Failed to read DEM with rasterio ({e}). Generating analytical Western Ghats profile.")

        return self.generate_western_ghats_profile()

    def generate_western_ghats_profile(self) -> np.ndarray:
        """
        Generates an analytical representation of the Western Ghats windward escarpment.
        Coastal plain (0-40 km eastward) rising steeply to 1500-1800 m MSL crest between 50-70 km.
        """
        nr, na = self.n_range_gates, self.n_azimuths
        r_mesh, az_mesh = np.meshgrid(self.ranges_km, self.azimuths_deg, indexing="ij")

        # Western Ghats orientation: runs approximately NNW to SSE (~160 deg).
        # Radials towards the East (azimuth 40 to 140 deg) cross the steep coastal escarpment.
        east_component = r_mesh * np.sin(np.radians(az_mesh))  # km eastward from radar

        # Ridge located 45 to 65 km East of coastal radars (e.g. Mumbai, Goa)
        ridge_distance_km = 50.0
        ridge_width_km = 12.0
        peak_elevation_m = 1750.0

        # Gaussian ridge profile + plateau to the east
        ridge = peak_elevation_m * np.exp(-((east_component - ridge_distance_km) ** 2) / (2.0 * (ridge_width_km**2)))
        deccan_plateau = 700.0 / (1.0 + np.exp(-(east_component - (ridge_distance_km + 15.0)) / 5.0))

        # Coastal plains and Arabian Sea (west of radar)
        elevation = np.where(east_component > 10.0, ridge + deccan_plateau, 20.0)
        # Sea westwards
        elevation = np.where(east_component <= 0.0, 0.0, elevation)

        # Add realistic terrain roughness
        roughness = np.sin(r_mesh * 0.5) * np.cos(az_mesh * 0.2) * 40.0
        elevation = np.maximum(elevation + roughness, 0.0)

        return elevation.astype(np.float32)
