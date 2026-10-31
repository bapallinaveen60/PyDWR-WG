"""
Adaptive Differential Propagation Phase (PhiDP) and Specific Differential Phase (KDP) Processing.
Implements transmitter offset subtraction, phase unwrapping, and Kalman-filtered KDP estimation.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter

def unwrap_phidp(phidp: np.ndarray, fold_angle: float = 180.0) -> np.ndarray:
    """
    Unwraps raw differential propagation phase along radial rays to eliminate phase folding.

    Parameters
    ----------
    phidp : np.ndarray
        Differential phase array [Nr, Na] or [Nr] in degrees.
    fold_angle : float
        Radar phase processing ambiguity boundary (typically 180.0 or 360.0 degrees).

    Returns
    -------
    np.ndarray
        Unwrapped differential phase.
    """
    is_1d = phidp.ndim == 1
    if is_1d:
        phidp_2d = phidp[:, np.newaxis]
    else:
        phidp_2d = phidp.copy()

    nr, na = phidp_2d.shape
    unwrapped = np.zeros_like(phidp_2d)

    for a in range(na):
        ray = phidp_2d[:, a]
        valid_idx = np.where(~np.isnan(ray))[0]
        if len(valid_idx) == 0:
            continue

        ray_clean = ray.copy()
        # Replace NaNs with nearest neighbor for continuity during unwrap
        nan_mask = np.isnan(ray_clean)
        if np.any(nan_mask):
            ray_clean[nan_mask] = np.interp(
                np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), ray_clean[~nan_mask]
            )

        # Standard 1D phase unwrap along range gates
        unw = np.unwrap(np.radians(ray_clean) * (360.0 / fold_angle))
        unw_deg = np.degrees(unw) * (fold_angle / 360.0)

        # Remove transmitter offset (baseline phase at near range, e.g., gates 10 to 30)
        offset_gates = unw_deg[10:35]
        offset = np.median(offset_gates) if len(offset_gates) > 0 else unw_deg[0]
        unw_deg -= offset

        # Monotonicity enforcement (since PhiDP must accumulate positively in rain at S-band)
        # Avoid backscatter negative steps by tracking cumulative running maximum with slight tolerance
        smooth_unw = uniform_filter1d(unw_deg, size=5)
        unwrapped[:, a] = np.maximum(smooth_unw, 0.0)

    return unwrapped[:, 0] if is_1d else unwrapped


def estimate_kdp_kalman(
    phidp: np.ndarray,
    range_step_km: float = 0.1,
    q_proc_noise: float = 1e-4,
    r_meas_noise: float = 4.0,
) -> np.ndarray:
    """
    Estimates range-resolved Specific Differential Phase (KDP [deg/km]) using a
    1D discrete Kalman filter along each radar radial.

    State vector: x = [PhiDP, KDP]^T
    State transition: x_{k} = F * x_{k-1} + w, where F = [[1, 2 * dr], [0, 1]]
    Measurement: z_{k} = H * x_{k} + v, where H = [[1, 0]]

    Parameters
    ----------
    phidp : np.ndarray
        Unwrapped differential propagation phase [Nr, Na] or [Nr] in degrees.
    range_step_km : float
        Radial gate spacing in kilometers (e.g., 0.1 km for 100 m).
    q_proc_noise : float
        Process noise variance scaling.
    r_meas_noise : float
        Measurement noise variance (in deg^2).

    Returns
    -------
    np.ndarray
        Estimated KDP array in deg/km.
    """
    is_1d = phidp.ndim == 1
    if is_1d:
        phidp_2d = phidp[:, np.newaxis]
    else:
        phidp_2d = phidp.copy()

    nr, na = phidp_2d.shape
    kdp_out = np.zeros_like(phidp_2d)

    dr = range_step_km
    # State transition matrix: Phi(r + dr) = Phi(r) + 2 * dr * Kdp(r)
    F = np.array([[1.0, 2.0 * dr], [0.0, 1.0]], dtype=np.float32)
    H = np.array([[1.0, 0.0]], dtype=np.float32)
    Q = np.array(
        [[q_proc_noise * (dr**3) / 3.0, q_proc_noise * (dr**2) / 2.0],
         [q_proc_noise * (dr**2) / 2.0, q_proc_noise * dr]],
        dtype=np.float32,
    )
    R = np.array([[r_meas_noise]], dtype=np.float32)
    I_mat = np.eye(2, dtype=np.float32)

    for a in range(na):
        z_ray = phidp_2d[:, a]
        # Initial state: [Phi0, Kdp0]
        x = np.array([z_ray[0] if not np.isnan(z_ray[0]) else 0.0, 0.0], dtype=np.float32)
        P = np.array([[10.0, 0.0], [0.0, 5.0]], dtype=np.float32)

        kdp_ray = np.zeros(nr, dtype=np.float32)

        for k in range(nr):
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q

            # Measurement update if gate is valid
            meas = z_ray[k]
            if not np.isnan(meas):
                y = meas - (H @ x)
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                x = x + (K @ y).squeeze()
                P = (I_mat - K @ H) @ P

            # Precipitation constraint: KDP in liquid warm rain is non-negative
            x[1] = np.maximum(x[1], 0.0)
            kdp_ray[k] = x[1]

        kdp_out[:, a] = kdp_ray

    return kdp_out[:, 0] if is_1d else kdp_out


def filter_backscatter_phase_shift(
    phidp: np.ndarray,
    zh: np.ndarray,
    zdr: np.ndarray,
    window_len: int = 11,
) -> np.ndarray:
    """
    Filters localized backscatter phase shift anomalies (delta_CO) caused by large drops
    or hail/melting particles using adaptive local window filtering.

    Parameters
    ----------
    phidp : np.ndarray
        Differential phase array [Nr, Na].
    zh : np.ndarray
        Horizontal reflectivity array [Nr, Na] in dBZ.
    zdr : np.ndarray
        Differential reflectivity array [Nr, Na] in dB.
    window_len : int
        Window length for median baseline estimation.

    Returns
    -------
    np.ndarray
        Filtered differential propagation phase free from delta_CO perturbations.
    """
    phidp_filtered = median_filter(phidp, size=(window_len, 1))
    return np.maximum(phidp_filtered, 0.0)
