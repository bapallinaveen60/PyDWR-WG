"""
Operational Baseline Clutter Mitigation Algorithms for Benchmarking.
Implements GMAP zero-Doppler notch filter, Gabella texture filter, and Fuzzy Logic CMD.
"""

import numpy as np
from scipy.ndimage import uniform_filter
from typing import Tuple

def gmap_notch_filter(
    zh: np.ndarray,
    vr: np.ndarray,
    notch_width_ms: float = 1.0,
    replacement_value: float = -10.0,
) -> np.ndarray:
    """
    Emulates operational Gaussian Model Adaptive Processing (GMAP) zero-Doppler notch filtering.
    Zero-velocity gates (|vr| < notch_width_ms) are assumed to be stationary ground clutter and removed.

    Note: This is the operational vulnerability causing monsoonal warm-rain signal erasure.
    """
    zh_filtered = zh.copy()
    notch_mask = np.abs(vr) < notch_width_ms
    zh_filtered[notch_mask] = replacement_value
    return zh_filtered


def gabella_texture_filter(
    zh: np.ndarray,
    texture_threshold_dbz: float = 12.0,
    window_size: int = 5,
) -> np.ndarray:
    """
    Gabella texture filter based on local reflectivity differences.
    Ground targets show high localized texture variance compared to smooth precipitation.
    """
    mean_zh = uniform_filter(zh, size=window_size)
    texture = np.abs(zh - mean_zh)

    clutter_mask = texture > texture_threshold_dbz
    zh_filtered = zh.copy()
    zh_filtered[clutter_mask] = -10.0
    return zh_filtered


def fuzzy_logic_cmd(
    zh: np.ndarray,
    rhohv: np.ndarray,
    vr: np.ndarray,
    sw: np.ndarray,
    clutter_threshold: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Operational Clutter Mitigation Decision (CMD) algorithm based on fuzzy logic membership.

    Membership functions:
    - MF_rhoHV: High probability of clutter when rhoHV < 0.85
    - MF_vr: High probability of clutter when |vr| is near 0
    - MF_sw: High probability of clutter when spectrum width is very narrow (< 0.6 m/s)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - zh_filtered : Reflectivity with clutter gates masked
        - clutter_probability : Fuzzy clutter membership score [0, 1]
    """
    # Membership for rhoHV: S-curve (clutter when rhoHV is low)
    mf_rho = np.clip((0.95 - rhohv) / (0.95 - 0.70), 0.0, 1.0)

    # Membership for Vr: Near zero
    mf_vr = np.exp(- (vr / 1.5) ** 2)

    # Membership for SW: Narrow spectrum width
    mf_sw = np.clip((1.5 - sw) / 1.0, 0.0, 1.0)

    # Weighted fuzzy score
    clutter_prob = 0.45 * mf_rho + 0.35 * mf_vr + 0.20 * mf_sw
    clutter_mask = clutter_prob >= clutter_threshold

    zh_filtered = zh.copy()
    zh_filtered[clutter_mask] = -10.0

    return zh_filtered, clutter_prob
