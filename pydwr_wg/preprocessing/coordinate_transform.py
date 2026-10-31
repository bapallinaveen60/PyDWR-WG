"""
Coordinate System Transformations and Polar-to-Cartesian Regridding.
Maintains radar native gate geometry and provides Cartesian interpolation for satellite integration.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional
from pydwr_wg.constants import EARTH_RADIUS_KM

def polar_to_cartesian_coords(
    ranges_km: np.ndarray,
    azimuths_deg: np.ndarray,
    radar_lat: float,
    radar_lon: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms polar radar gate coordinates to planar Cartesian (x, y) [km]
    and geographic (lat, lon) coordinates.

    Parameters
    ----------
    ranges_km : np.ndarray
        1D array of slant/ground ranges [Nr].
    azimuths_deg : np.ndarray
        1D array of azimuth angles clockwise from True North [Na].
    radar_lat : float
        Latitude of the radar transmitter in decimal degrees.
    radar_lon : float
        Longitude of the radar transmitter in decimal degrees.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - x_km : 2D array [Nr, Na]
        - y_km : 2D array [Nr, Na]
        - lat : 2D array [Nr, Na]
        - lon : 2D array [Nr, Na]
    """
    r_mesh, az_mesh = np.meshgrid(ranges_km, azimuths_deg, indexing="ij")
    az_rad = np.radians(az_mesh)

    # Meteorological azimuth: 0 deg = North, 90 deg = East
    x_km = r_mesh * np.sin(az_rad)
    y_km = r_mesh * np.cos(az_rad)

    # Spherical approximation for local region
    km_per_deg_lat = 111.132
    km_per_deg_lon = 111.320 * np.cos(np.radians(radar_lat))

    lat = radar_lat + (y_km / km_per_deg_lat)
    lon = radar_lon + (x_km / km_per_deg_lon)

    return x_km, y_km, lat, lon


def polar_to_cartesian_grid(
    polar_field: np.ndarray,
    range_step_km: float = 0.1,
    azimuth_step_deg: float = 1.0,
    grid_extent_km: float = 100.0,
    grid_resolution_km: float = 1.0,
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolates a 2D polar radar array [Nr, Na] onto a regular Cartesian grid [Ny, Nx].

    Parameters
    ----------
    polar_field : np.ndarray
        2D polar array [Nr, Na].
    range_step_km : float
        Radial gate spacing in km.
    azimuth_step_deg : float
        Azimuthal ray spacing in degrees.
    grid_extent_km : float
        Half-width of output Cartesian grid (e.g. 100 km -> [-100, +100] km).
    grid_resolution_km : float
        Grid cell size in km (e.g. 1.0 km).
    fill_value : float
        Value to fill outside radar observable range.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - cartesian_grid : 2D array [Ny, Nx]
        - x_vec : 1D array of x coordinates [km]
        - y_vec : 1D array of y coordinates [km]
    """
    nr, na = polar_field.shape
    r_coords = np.arange(1, nr + 1) * range_step_km
    az_coords = np.arange(0, na) * azimuth_step_deg

    # Azimuth wrap for smooth circular interpolation (add 360 deg)
    az_extended = np.append(az_coords, 360.0)
    field_extended = np.hstack([polar_field, polar_field[:, :1]])

    interp_func = RegularGridInterpolator(
        (r_coords, az_extended),
        field_extended,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )

    x_vec = np.arange(-grid_extent_km, grid_extent_km + grid_resolution_km, grid_resolution_km)
    y_vec = np.arange(-grid_extent_km, grid_extent_km + grid_resolution_km, grid_resolution_km)
    gx, gy = np.meshgrid(x_vec, y_vec)

    # Compute target r and azimuth
    target_r = np.sqrt(gx**2 + gy**2)
    # Target azimuth clockwise from North [0, 360)
    target_az = np.degrees(np.arctan2(gx, gy)) % 360.0

    target_pts = np.stack([target_r.ravel(), target_az.ravel()], axis=-1)
    cartesian_grid = interp_func(target_pts).reshape(gx.shape)

    # Mask points beyond maximum radar range
    max_range = nr * range_step_km
    cartesian_grid[target_r > max_range] = fill_value

    return cartesian_grid, x_vec, y_vec
