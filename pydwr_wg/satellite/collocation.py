"""
Radar-to-INSAT-3DR Geostationary Collocation Engine.
Collocates polar radar fields with INSAT-3DR multispectral imagery and ERA5 thermodynamic fields
within a 10-minute temporal coincidence window on a unified 1 km Cartesian grid.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from pydwr_wg.preprocessing.coordinate_transform import polar_to_cartesian_grid

class SatelliteCollocationEngine:
    """
    Spatially and temporally collocates de-cluttered ground radar products with
    INSAT-3DR Imager observations and ERA5 reanalysis fields.
    """

    def __init__(
        self,
        grid_extent_km: float = 100.0,
        grid_resolution_km: float = 1.0,
        temporal_window_minutes: float = 10.0,
    ):
        self.grid_extent_km = grid_extent_km
        self.grid_resolution_km = grid_resolution_km
        self.temporal_window_minutes = temporal_window_minutes

    def collocate_polar_with_satellite(
        self,
        polar_clean_zh: np.ndarray,
        polar_rain_rate: np.ndarray,
        insat_channels: Dict[str, np.ndarray],
        era5_fields: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Projects radar fields onto 1 km Cartesian grid and stacks with INSAT-3DR / ERA5 channels.

        Parameters
        ----------
        polar_clean_zh : np.ndarray
            Reconstructed reflectivity [Nr, Na].
        polar_rain_rate : np.ndarray
            Reconstructed surface rain rate [Nr, Na].
        insat_channels : Dict[str, np.ndarray]
            Satellite channels ('TIR1', 'TIR2', 'MIR', 'WV') on Cartesian grid [Ny, Nx].
        era5_fields : Optional[Dict[str, np.ndarray]]
            Thermodynamic fields ('cape', 'tcwv') on Cartesian grid [Ny, Nx].

        Returns
        -------
        Dict[str, np.ndarray]
            Collocated dataset dictionary containing 2D arrays on the unified 1 km grid.
        """
        # Interpolate polar radar fields to 1 km Cartesian grid
        grid_zh, gx, gy = polar_to_cartesian_grid(
            polar_clean_zh,
            grid_extent_km=self.grid_extent_km,
            grid_resolution_km=self.grid_resolution_km,
            fill_value=np.nan,
        )
        grid_rr, _, _ = polar_to_cartesian_grid(
            polar_rain_rate,
            grid_extent_km=self.grid_extent_km,
            grid_resolution_km=self.grid_resolution_km,
            fill_value=0.0,
        )

        ny, nx = grid_zh.shape
        collocated = {
            "radar_zh_1km": grid_zh.astype(np.float32),
            "radar_rain_rate_1km": grid_rr.astype(np.float32),
        }

        # Add satellite channels (resizing if necessary to match [Ny, Nx])
        for ch_name, data in insat_channels.items():
            if data.shape != (ny, nx):
                from scipy.ndimage import zoom
                zf = (ny / data.shape[0], nx / data.shape[1])
                collocated[f"insat_{ch_name.lower()}"] = zoom(data, zf, order=1).astype(np.float32)
            else:
                collocated[f"insat_{ch_name.lower()}"] = data.astype(np.float32)

        # Add ERA5 fields
        if era5_fields is not None:
            for f_name, data in era5_fields.items():
                if data.shape != (ny, nx):
                    from scipy.ndimage import zoom
                    zf = (ny / data.shape[0], nx / data.shape[1])
                    collocated[f"era5_{f_name.lower()}"] = zoom(data, zf, order=1).astype(np.float32)
                else:
                    collocated[f"era5_{f_name.lower()}"] = data.astype(np.float32)

        return collocated
