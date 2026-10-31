import unittest
import numpy as np
from pydwr_wg.physics.self_consistency import PolarimetricSelfConsistency
from pydwr_wg.physics.qpe_estimators import (
    r_marshall_palmer,
    r_western_ghats_warm,
    r_kdp_sband,
    r_zh_zdr_sband,
    adaptive_polarimetric_qpe,
)
from pydwr_wg.physics.microphysics import WarmRainMicrophysics

class TestQPEAndPhysics(unittest.TestCase):
    def test_self_consistency(self):
        sc = PolarimetricSelfConsistency(radar_band="S")
        kdp = np.array([0.1, 0.5, 2.0])
        zdr = np.array([0.3, 0.8, 1.5])
        zh_hat = sc.estimate_zh_from_polarimetry(kdp, zdr)

        self.assertTrue(np.all(np.diff(zh_hat) > 0))

        zh_clutter = np.array([65.0, 70.0])
        zdr_clutter = np.array([-2.0, 5.0])
        kdp_clutter = np.array([0.01, 0.02])
        rhohv_clutter = np.array([0.45, 0.55])

        delta_z, score = sc.compute_clutter_discrepancy(zh_clutter, zdr_clutter, kdp_clutter, rhohv_clutter)
        self.assertTrue(np.all(delta_z > 25.0))
        self.assertTrue(np.all(score < 0.1))

    def test_qpe_estimators(self):
        zh = np.array([25.0, 35.0, 45.0])
        r_mp = r_marshall_palmer(zh)
        r_wg = r_western_ghats_warm(zh)

        self.assertTrue(np.all(r_wg > 0.0))
        self.assertTrue(np.all(np.diff(r_mp) > 0))

        kdp = np.array([0.2, 1.0, 3.0])
        r_kdp = r_kdp_sband(kdp)
        self.assertTrue(np.all(r_kdp > 0.0))

        bbf = np.array([0.0, 0.4, 0.8])
        zdr = np.array([0.5, 1.0, 1.5])
        r_adapt = adaptive_polarimetric_qpe(zh, zdr, kdp, bbf)
        self.assertEqual(r_adapt.shape, (3,))
        self.assertTrue(np.all(r_adapt >= 0.0))

    def test_microphysics(self):
        zdr = np.array([0.2, 0.8, 1.4])
        dm = WarmRainMicrophysics.estimate_dm_from_polarimetry(zdr)
        self.assertTrue(np.all((dm >= 1.0) & (dm <= 1.8)))

if __name__ == "__main__":
    unittest.main()
