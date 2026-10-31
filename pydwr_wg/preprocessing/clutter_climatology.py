"""
Dry-Weather Clutter Climatology Compositing and Baseline Mask Generation.
Processes pre-monsoon dry scans to construct stationary terrain clutter and main-bang probability maps.
"""

import numpy as np
from typing import Dict, Any, List, Optional

def build_clutter_climatology(
    dry_scans_zh: List[np.ndarray],
    dry_scans_rhohv: List[np.ndarray],
    dry_scans_zdr: Optional[List[np.ndarray]] = None,
    zh_threshold_dbz: float = 15.0,
    clutter_frequency_cutoff: float = 0.50,
) -> Dict[str, np.ndarray]:
    """
    Constructs stationary terrain clutter probability maps from dry-season radar volume scans.

    Parameters
    ----------
    dry_scans_zh : List[np.ndarray]
        List of 2D polar reflectivity arrays [Nr, Na] recorded under dry-air conditions.
    dry_scans_rhohv : List[np.ndarray]
        List of 2D polar copolar correlation arrays [Nr, Na].
    dry_scans_zdr : Optional[List[np.ndarray]]
        List of 2D polar differential reflectivity arrays [Nr, Na].
    zh_threshold_dbz : float
        Minimum reflectivity threshold to register an echo.
    clutter_frequency_cutoff : float
        Frequency threshold above which a gate is classified as persistent clutter.

    Returns
    -------
    Dict[str, np.ndarray]
        - clutter_frequency: Clutter frequency [0.0, 1.0]
        - clutter_mask: Binary mask of persistent ground clutter
        - mean_zh: Mean clutter reflectivity
        - mean_rhohv: Mean clutter copolar correlation
    """
    n_scans = len(dry_scans_zh)
    if n_scans == 0:
        raise ValueError("dry_scans_zh list cannot be empty.")

    nr, na = dry_scans_zh[0].shape
    echo_count = np.zeros((nr, na), dtype=np.float32)
    zh_sum = np.zeros((nr, na), dtype=np.float32)
    rhohv_sum = np.zeros((nr, na), dtype=np.float32)

    for i in range(n_scans):
        zh = dry_scans_zh[i]
        rho = dry_scans_rhohv[i]

        valid_echo = (zh >= zh_threshold_dbz) & (~np.isnan(zh))
        echo_count += valid_echo.astype(np.float32)
        zh_sum += np.where(valid_echo, zh, 0.0)
        rhohv_sum += np.where(valid_echo, rho, 0.0)

    clutter_freq = echo_count / float(n_scans)
    mean_zh = np.divide(zh_sum, echo_count, out=np.zeros_like(zh_sum), where=echo_count > 0)
    mean_rhohv = np.divide(rhohv_sum, echo_count, out=np.zeros_like(rhohv_sum), where=echo_count > 0)

    # Clutter mask: persistent echoes with non-meteorological correlation (rhoHV < 0.85)
    clutter_mask = (clutter_freq >= clutter_frequency_cutoff) & (mean_rhohv < 0.85)

    return {
        "clutter_frequency": clutter_freq,
        "clutter_mask": clutter_mask.astype(np.uint8),
        "mean_zh": mean_zh,
        "mean_rhohv": mean_rhohv,
    }


def generate_main_bang_mask(
    nr: int = 1000,
    na: int = 360,
    blind_zone_km: float = 5.0,
    range_step_m: float = 100.0,
) -> np.ndarray:
    """
    Creates a binary mask identifying the radar main-bang transmit-receive transient zone.

    Parameters
    ----------
    nr : int
        Number of range gates.
    na : int
        Number of azimuth rays.
    blind_zone_km : float
        Radius of transmitter saturation zone in kilometers.
    range_step_m : float
        Gate spacing in meters.

    Returns
    -------
    np.ndarray
        Binary mask [Nr, Na] where 1 indicates main-bang corrupted gates.
    """
    blind_gates = int(np.ceil((blind_zone_km * 1000.0) / range_step_m))
    mask = np.zeros((nr, na), dtype=np.uint8)
    mask[:blind_gates, :] = 1
    return mask
