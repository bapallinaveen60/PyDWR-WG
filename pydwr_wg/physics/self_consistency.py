"""
Polarimetric Self-Consistency Relationships for Warm Monsoonal Rain.
Evaluates microphysical coherence between ZH, ZDR, and KDP to discriminate non-meteorological terrain clutter.
"""

import numpy as np
from typing import Tuple

class PolarimetricSelfConsistency:
    """
    Evaluates empirical self-consistency relations between radar horizontal reflectivity (ZH),
    differential reflectivity (ZDR), and specific differential phase (KDP).
    """

    def __init__(self, radar_band: str = "S"):
        self.radar_band = radar_band.upper()
        # Polynomial coefficients for S-band monsoonal warm rain:
        # log10(Kdp) = b0 + b1 * ZH + b2 * ZDR + b3 * ZDR^2
        if self.radar_band == "S":
            self.a0 = 38.4
            self.a1 = 12.6
            self.a2 = 4.25
        else:  # X-band
            self.a0 = 34.1
            self.a1 = 11.2
            self.a2 = 3.8

    def estimate_zh_from_polarimetry(
        self,
        kdp: np.ndarray,
        zdr: np.ndarray,
    ) -> np.ndarray:
        """
        Estimates expected horizontal reflectivity ZH [dBZ] from KDP and ZDR
        assuming a liquid precipitation medium.

        ZH_hat = a0 + a1 * log10(max(KDP, 1e-3)) + a2 * ZDR

        Parameters
        ----------
        kdp : np.ndarray
            Specific differential phase in deg/km.
        zdr : np.ndarray
            Differential reflectivity in dB.

        Returns
        -------
        np.ndarray
            Microphysically consistent reflectivity field in dBZ.
        """
        kdp_safe = np.maximum(kdp, 1e-3)
        zh_hat = self.a0 + self.a1 * np.log10(kdp_safe) + self.a2 * np.clip(zdr, 0.0, 4.0)
        return zh_hat

    def compute_clutter_discrepancy(
        self,
        zh: np.ndarray,
        zdr: np.ndarray,
        kdp: np.ndarray,
        rhohv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the self-consistency violation metric delta_Z.
        Terrain clutter exhibits high delta_Z (> 20-30 dB) and low rhoHV (< 0.85).

        Parameters
        ----------
        zh : np.ndarray
            Measured reflectivity [dBZ].
        zdr : np.ndarray
            Measured differential reflectivity [dB].
        kdp : np.ndarray
            Measured specific differential phase [deg/km].
        rhohv : np.ndarray
            Measured copolar correlation coefficient.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - delta_z: Reflectivity self-consistency residual [dB]
            - consistency_score: Normalised precipitation fidelity score [0.0, 1.0]
        """
        zh_hat = self.estimate_zh_from_polarimetry(kdp, zdr)
        delta_z = np.abs(zh - zh_hat)

        # Meteorological score: high rhoHV, low delta_Z, positive KDP
        # Pure rain: delta_z < 6 dB and rhoHV > 0.95
        score = np.exp(- (delta_z / 12.0) ** 2) * np.clip((rhohv - 0.70) / 0.28, 0.0, 1.0)

        return delta_z, score.astype(np.float32)
