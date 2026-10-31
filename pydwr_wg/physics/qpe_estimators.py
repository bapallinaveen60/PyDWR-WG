"""
Quantitative Precipitation Estimation (QPE) Formulations for Tropical Orographic Warm Rain.
Provides attenuation- and blockage-invariant polarimetric estimators tailored for S-band Western Ghats regimes.
"""

import numpy as np

def r_marshall_palmer(zh_dbz: np.ndarray) -> np.ndarray:
    """
    Standard Marshall-Palmer Z-R relation: Z = 200 * R^1.6
    R = (10^(ZH / 10) / 200)^(1 / 1.6) [mm/h]
    """
    z_lin = 10.0 ** (zh_dbz / 10.0)
    r = (np.maximum(z_lin, 0.0) / 200.0) ** (1.0 / 1.6)
    return np.where(zh_dbz > 5.0, r, 0.0).astype(np.float32)


def r_western_ghats_warm(zh_dbz: np.ndarray) -> np.ndarray:
    """
    Western Ghats localized warm-rain Z-R relation (HACPL calibrated):
    Z = 165 * R^1.38
    """
    z_lin = 10.0 ** (zh_dbz / 10.0)
    r = (np.maximum(z_lin, 0.0) / 165.0) ** (1.0 / 1.38)
    return np.where(zh_dbz > 5.0, r, 0.0).astype(np.float32)


def r_kdp_sband(kdp: np.ndarray) -> np.ndarray:
    """
    Specific differential phase rain rate estimator at S-band:
    R(Kdp) = 50.7 * |Kdp|^0.85 [mm/h]
    Completely invariant to partial beam blockage, receiver miscalibration, and isotropic attenuation.
    """
    kdp_pos = np.maximum(kdp, 0.0)
    r = 50.7 * (kdp_pos ** 0.85)
    return r.astype(np.float32)


def r_zh_zdr_sband(zh_dbz: np.ndarray, zdr_db: np.ndarray) -> np.ndarray:
    """
    Polarimetric reflectivity and differential reflectivity estimator:
    R(ZH, ZDR) = 0.0076 * (10^(0.093 * ZH)) * (10^(-0.281 * ZDR)) [mm/h]
    """
    zdr_clipped = np.clip(zdr_db, 0.1, 4.0)
    r = 0.0076 * (10.0 ** (0.093 * zh_dbz)) * (10.0 ** (-0.281 * zdr_clipped))
    return np.where(zh_dbz > 5.0, r, 0.0).astype(np.float32)


def r_kdp_zdr_sband(kdp: np.ndarray, zdr_db: np.ndarray) -> np.ndarray:
    """
    Polarimetric phase and shape estimator (unbiased by beam blockage):
    R(Kdp, ZDR) = 90.8 * (Kdp^0.93) * (10^(-0.169 * ZDR)) [mm/h]
    """
    kdp_pos = np.maximum(kdp, 0.0)
    zdr_clipped = np.clip(zdr_db, 0.1, 4.0)
    r = 90.8 * (kdp_pos ** 0.93) * (10.0 ** (-0.169 * zdr_clipped))
    return r.astype(np.float32)


def adaptive_polarimetric_qpe(
    zh_dbz: np.ndarray,
    zdr_db: np.ndarray,
    kdp: np.ndarray,
    bbf: np.ndarray,
) -> np.ndarray:
    """
    Adaptive operational decision tree for optimal surface rainfall rate estimation:
    - If beam blockage BBF > 0.20: rely strictly on R(Kdp) to prevent underestimation.
    - If Kdp >= 0.30 deg/km (moderate to heavy rain): rely on R(Kdp, ZDR).
    - If Kdp < 0.30 deg/km and unblocked: rely on R(ZH, ZDR) or warm Z-R.

    Parameters
    ----------
    zh_dbz : np.ndarray
        Reflectivity [Nr, Na].
    zdr_db : np.ndarray
        Differential reflectivity [Nr, Na].
    kdp : np.ndarray
        Specific differential phase [Nr, Na].
    bbf : np.ndarray
        Partial beam blockage fraction [Nr, Na].

    Returns
    -------
    np.ndarray
        Surface rain rate array [Nr, Na] in mm/h.
    """
    r_kdp_val = r_kdp_sband(kdp)
    r_kdp_zdr_val = r_kdp_zdr_sband(kdp, zdr_db)
    r_zh_zdr_val = r_zh_zdr_sband(zh_dbz, zdr_db)
    r_warm_val = r_western_ghats_warm(zh_dbz)

    # Initialize with warm-rain Z-R
    r_final = r_warm_val.copy()

    # Moderate rain with reliable phase
    use_kdp_zdr = (kdp >= 0.25) & (bbf < 0.50)
    r_final = np.where(use_kdp_zdr, r_kdp_zdr_val, r_final)

    # Blocked sectors (BBF >= 0.20): switch to pure R(KDP)
    use_pure_kdp = (bbf >= 0.20) & (kdp > 0.05)
    r_final = np.where(use_pure_kdp, r_kdp_val, r_final)

    # Light drizzle in clear sectors
    use_zh_zdr = (kdp < 0.25) & (bbf < 0.20) & (zh_dbz > 15.0)
    r_final = np.where(use_zh_zdr, r_zh_zdr_val, r_final)

    return np.maximum(r_final, 0.0).astype(np.float32)
