import unittest
import numpy as np
from pydwr_wg.preprocessing.phase_processing import unwrap_phidp, estimate_kdp_kalman

class TestPhaseProcessing(unittest.TestCase):
    def test_phase_unwrap(self):
        nr = 100
        true_phi = np.linspace(20, 250, nr)
        folded_phi = (true_phi + 180) % 180.0

        unwrapped = unwrap_phidp(folded_phi, fold_angle=180.0)
        self.assertEqual(unwrapped.shape, (nr,))
        self.assertTrue(np.all(unwrapped >= 0.0))

    def test_kalman_kdp(self):
        nr = 100
        dr = 0.1
        true_kdp = 1.5
        phidp = np.cumsum(2.0 * true_kdp * dr * np.ones(nr))
        noisy_phidp = phidp + np.random.normal(0, 0.2, nr)

        est_kdp = estimate_kdp_kalman(noisy_phidp, range_step_km=dr)
        self.assertEqual(est_kdp.shape, (nr,))
        self.assertTrue(0.8 < np.mean(est_kdp[20:]) < 2.2)
        self.assertTrue(np.all(est_kdp >= 0.0))

if __name__ == "__main__":
    unittest.main()
