"""
Semi-Synthetic Clutter Injection Pipeline (SCIP).
Superimposes dry-season clutter and virtual mountain blockage onto pristine Arabian Sea marine rain sweeps.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pydwr_wg.constants import CHANNEL_BOUNDS
from pydwr_wg.physics.qpe_estimators import r_western_ghats_warm

class SCIPPipeline:
    """
    Constructs paired training sets without ground-truth leakage by injecting
    terrain blockage and non-meteorological clutter into pristine radar observations.
    """

    def __init__(
        self,
        n_range_gates: int = 1000,
        n_azimuths: int = 360,
        range_step_m: float = 100.0,
    ):
        self.n_range_gates = n_range_gates
        self.n_azimuths = n_azimuths
        self.range_step_m = range_step_m

    def inject_clutter_and_blockage(
        self,
        pristine_sweep: Dict[str, np.ndarray],
        dem_polar: np.ndarray,
        bbf_polar: np.ndarray,
        h_beam_agl: np.ndarray,
        clutter_profile: Optional[Dict[str, np.ndarray]] = None,
        inject_main_bang: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesizes a corrupted 10-channel input tensor and paired ground truth targets.

        Parameters
        ----------
        pristine_sweep : Dict[str, np.ndarray]
            Pristine meteorological fields ('ZH', 'ZDR', 'RHOHV', 'PHIDP', 'KDP', 'VR', 'SW').
        dem_polar : np.ndarray
            Topography [Nr, Na] in meters MSL.
        bbf_polar : np.ndarray
            Partial beam blockage fraction [Nr, Na] in [0, 1].
        h_beam_agl : np.ndarray
            Center beam height AGL [Nr, Na] in meters.
        clutter_profile : Optional[Dict[str, np.ndarray]]
            Pre-monsoon dry clutter observations.
        inject_main_bang : bool
            Whether to inject transmit-receive switching saturation in range 0-5 km.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            - X_tensor : np.ndarray [10, Nr, Na] (normalized input)
            - M_clutter : np.ndarray [Nr, Na] (target clutter mask)
            - ZH_clean : np.ndarray [Nr, Na] (target clean reflectivity dBZ)
            - R_clean : np.ndarray [Nr, Na] (target surface rain rate mm/h)
        """
        nr, na = self.n_range_gates, self.n_azimuths

        # 1. Clean ground-truth fields
        zh_true = pristine_sweep["ZH"].copy()
        zdr_true = pristine_sweep["ZDR"].copy()
        rhohv_true = pristine_sweep["RHOHV"].copy()
        phidp_true = pristine_sweep["PHIDP"].copy()
        kdp_true = pristine_sweep["KDP"].copy()
        vr_true = pristine_sweep["VR"].copy()
        sw_true = pristine_sweep["SW"].copy()

        # Compute ground-truth surface rain rate
        r_clean = r_western_ghats_warm(zh_true)

        # 2. Corrupt with Partial Beam Blockage (PBB)
        # Power attenuation = 10 * log10(1 - BBF)
        # Power drops by ~3.98 dB at 60% blockage
        unblocked_fraction = np.clip(1.0 - bbf_polar, 1e-4, 1.0)
        power_loss_db = 10.0 * np.log10(unblocked_fraction)

        zh_corrupted = zh_true + power_loss_db
        zdr_corrupted = zdr_true + 0.3 * power_loss_db  # differential blockage effect

        # 3. Inject Terrain Ground Clutter
        m_clutter = np.zeros((nr, na), dtype=np.float32)

        if clutter_profile is not None:
            clutter_mask = clutter_profile["clutter_mask"] > 0
            zh_clutter = clutter_profile["mean_zh"]
            rho_clutter = clutter_profile["mean_rhohv"]
        else:
            # Analytical clutter injection where DEM is steep and beam intercepts ridge
            clutter_mask = (dem_polar > 800.0) & (h_beam_agl < 300.0) & (bbf_polar > 0.15)
            zh_clutter = np.random.uniform(50.0, 68.0, size=(nr, na))
            rho_clutter = np.random.uniform(0.35, 0.75, size=(nr, na))

        # Superimpose clutter backscatter power
        zh_lin_rain = 10.0 ** (zh_corrupted / 10.0)
        zh_lin_clut = 10.0 ** (zh_clutter / 10.0)
        zh_combined = 10.0 * np.log10(zh_lin_rain + zh_lin_clut)

        zh_corrupted = np.where(clutter_mask, zh_combined, zh_corrupted)
        rhohv_corrupted = np.where(clutter_mask, rho_clutter, rhohv_true)
        zdr_corrupted = np.where(clutter_mask, np.random.uniform(-4.0, 6.0, (nr, na)), zdr_corrupted)
        vr_corrupted = np.where(clutter_mask, np.random.normal(0.0, 0.15, (nr, na)), vr_true)
        sw_corrupted = np.where(clutter_mask, np.random.uniform(0.1, 0.6, (nr, na)), sw_true)
        m_clutter = np.where(clutter_mask, 1.0, m_clutter)

        # 4. Inject Main-Bang Blind Zone Transients (0 to 5 km)
        if inject_main_bang:
            main_bang_gates = int(5000.0 / self.range_step_m)
            mb_slice = slice(0, main_bang_gates)
            zh_corrupted[mb_slice, :] = np.random.uniform(58.0, 72.0, (main_bang_gates, na))
            rhohv_corrupted[mb_slice, :] = np.random.uniform(0.15, 0.38, (main_bang_gates, na))
            zdr_corrupted[mb_slice, :] = np.random.normal(0.0, 2.5, (main_bang_gates, na))
            m_clutter[mb_slice, :] = 1.0

        # Note: PhiDP downstream remains invariant to blockage!
        phidp_corrupted = phidp_true.copy()
        kdp_corrupted = kdp_true.copy()

        # 5. Normalize 10 channels to [0, 1] range for PTA-InNet
        c0 = self._norm(zh_corrupted, CHANNEL_BOUNDS["ZH"]["min"], CHANNEL_BOUNDS["ZH"]["max"])
        c1 = self._norm(zdr_corrupted, CHANNEL_BOUNDS["ZDR"]["min"], CHANNEL_BOUNDS["ZDR"]["max"])
        c2 = self._norm(np.clip(rhohv_corrupted, 0.2, 1.0), 0.2, 1.0)
        c3 = self._norm(phidp_corrupted, CHANNEL_BOUNDS["PHIDP"]["min"], CHANNEL_BOUNDS["PHIDP"]["max"])
        c4 = self._norm(kdp_corrupted, CHANNEL_BOUNDS["KDP"]["min"], CHANNEL_BOUNDS["KDP"]["max"])
        c5 = self._norm(vr_corrupted, CHANNEL_BOUNDS["VR"]["min"], CHANNEL_BOUNDS["VR"]["max"])
        c6 = self._norm(sw_corrupted, CHANNEL_BOUNDS["SW"]["min"], CHANNEL_BOUNDS["SW"]["max"])
        c7 = self._norm(dem_polar, CHANNEL_BOUNDS["DEM"]["min"], CHANNEL_BOUNDS["DEM"]["max"])
        c8 = np.clip(bbf_polar, 0.0, 1.0)
        c9 = self._norm(h_beam_agl, CHANNEL_BOUNDS["HBEAM"]["min"], CHANNEL_BOUNDS["HBEAM"]["max"])

        x_tensor = np.stack([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9], axis=0).astype(np.float32)

        return x_tensor, m_clutter.astype(np.float32), zh_true.astype(np.float32), r_clean.astype(np.float32)

    @staticmethod
    def _norm(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return np.clip((arr - min_val) / (max_val - min_val), 0.0, 1.0).astype(np.float32)
