"""
3D Ray-tracing and Topographic Beam Blockage Computation under Complex Terrain.
Implements the 4/3 effective Earth radius refraction model and Gaussian beam integration.
"""

import numpy as np
from typing import Tuple, Optional
from pydwr_wg.constants import EARTH_RADIUS_KM, EFFECTIVE_EARTH_FACTOR

def compute_beam_height(
    range_km: np.ndarray,
    elevation_deg: float,
    radar_altitude_m: float = 0.0,
    ke: float = EFFECTIVE_EARTH_FACTOR,
    re_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """
    Computes beam centerline height above Mean Sea Level (MSL) using the 4/3 Earth radius model.

    h(r) = sqrt(r^2 + (ke*Re)^2 + 2 * r * ke * Re * sin(theta_e)) - ke*Re + h_radar

    Parameters
    ----------
    range_km : np.ndarray
        Slant range from radar transmitter in kilometers.
    elevation_deg : float
        Antenna pointing elevation angle in degrees.
    radar_altitude_m : float, optional
        Altitude of the radar feedhorn above MSL in meters, by default 0.0.
    ke : float, optional
        Effective Earth radius factor (standard atmosphere = 4/3), by default 4/3.
    re_km : float, optional
        Mean Earth radius in km, by default 6371.0 km.

    Returns
    -------
    np.ndarray
        Beam centerline height in meters above MSL.
    """
    theta_rad = np.radians(elevation_deg)
    r_eff_km = ke * re_km
    h_radar_km = radar_altitude_m / 1000.0

    # Slant range geometry with effective spherical earth curvature
    term1 = range_km**2 + (r_eff_km + h_radar_km)**2
    term2 = 2.0 * range_km * (r_eff_km + h_radar_km) * np.sin(theta_rad)
    h_beam_km = np.sqrt(term1 + term2) - r_eff_km

    return h_beam_km * 1000.0  # Convert to meters


def compute_ground_range(
    range_km: np.ndarray,
    elevation_deg: float,
    radar_altitude_m: float = 0.0,
    ke: float = EFFECTIVE_EARTH_FACTOR,
    re_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """
    Computes ground distance (geodesic arc length) along the Earth's surface.

    Parameters
    ----------
    range_km : np.ndarray
        Slant range in kilometers.
    elevation_deg : float
        Elevation angle in degrees.
    radar_altitude_m : float
        Radar altitude in meters.
    ke : float
        Effective Earth factor.
    re_km : float
        Earth radius in km.

    Returns
    -------
    np.ndarray
        Ground range in kilometers.
    """
    theta_rad = np.radians(elevation_deg)
    r_eff_km = ke * re_km
    h_beam_m = compute_beam_height(range_km, elevation_deg, radar_altitude_m, ke, re_km)
    h_beam_km = h_beam_m / 1000.0

    sin_psi = (range_km * np.cos(theta_rad)) / (r_eff_km + h_beam_km)
    sin_psi = np.clip(sin_psi, -1.0, 1.0)
    psi = np.arcsin(sin_psi)

    return r_eff_km * psi


def compute_partial_beam_blockage(
    dem_polar: np.ndarray,
    elevation_deg: float,
    radar_altitude_m: float = 60.0,
    beamwidth_deg: float = 1.0,
    range_step_m: float = 100.0,
    n_subbeams: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the cumulative Partial Beam Blockage Fraction (BBF) and beam centerline height
    above ground level (AGL) across a 2D polar array [Nr, Na].

    Parameters
    ----------
    dem_polar : np.ndarray
        Digital Elevation Model sampled onto the radar's polar coordinate grid [Nr, Na] in meters MSL.
    elevation_deg : float
        Radar tilt elevation angle in degrees.
    radar_altitude_m : float
        Radar altitude above MSL in meters.
    beamwidth_deg : float
        3-dB one-way antenna beamwidth in degrees (typically 1.0 deg for S-band).
    range_step_m : float
        Radial gate spacing in meters.
    n_subbeams : int
        Number of vertical Gaussian sub-beam integration slices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - bbf_cumulative : np.ndarray [Nr, Na]
            Cumulative partial beam blockage fraction in [0.0, 1.0].
        - h_beam_agl : np.ndarray [Nr, Na]
            Center beam height above ground level (AGL) in meters.
    """
    nr, na = dem_polar.shape
    range_km = (np.arange(1, nr + 1) * range_step_m) / 1000.0

    # 1. Centerline height above MSL across range
    h_center_msl = compute_beam_height(range_km, elevation_deg, radar_altitude_m)
    # Broadcast to [Nr, Na]
    h_center_2d = np.repeat(h_center_msl[:, np.newaxis], na, axis=1)

    # Beam height above ground level (clipped to min 0)
    h_beam_agl = np.maximum(h_center_2d - dem_polar, 0.0)

    # 2. Gaussian power integration across vertical sub-beams
    # Beam profile: f(y) = exp(-4 * ln(2) * (theta_v / beamwidth)^2)
    subbeam_angles = np.linspace(-beamwidth_deg, beamwidth_deg, n_subbeams)
    # Gaussian weights
    weights = np.exp(-4.0 * np.log(2.0) * (subbeam_angles / beamwidth_deg) ** 2)
    weights /= np.sum(weights)

    # Preallocate sub-beam heights [n_subbeams, Nr]
    subbeam_heights = np.zeros((n_subbeams, nr), dtype=np.float32)
    for i, offset in enumerate(subbeam_angles):
        subbeam_heights[i] = compute_beam_height(
            range_km, elevation_deg + offset, radar_altitude_m
        )

    # Evaluate blockage along radials
    bbf_cumulative = np.zeros((nr, na), dtype=np.float32)

    for a in range(na):
        terrain_radial = dem_polar[:, a]  # [Nr]
        # For each subbeam, check where terrain exceeds subbeam height
        # Shape: [n_subbeams, Nr]
        blocked = terrain_radial[np.newaxis, :] >= subbeam_heights
        # Weighted power blocked at each gate
        local_bbf = np.sum(blocked * weights[:, np.newaxis], axis=0)  # [Nr]

        # Cumulative blockage along the beam path cannot decrease down-range
        cum_block = np.maximum.accumulate(local_bbf)
        bbf_cumulative[:, a] = np.clip(cum_block, 0.0, 1.0)

    return bbf_cumulative, h_beam_agl
