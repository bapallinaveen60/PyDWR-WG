import unittest
import numpy as np
from pydwr_wg.evaluation.metrics import (
    compute_categorical_metrics,
    compute_low_velocity_rain_preservation,
    compute_reconstruction_fidelity,
    compute_qpe_metrics,
)

class TestMetrics(unittest.TestCase):
    def test_categorical_metrics(self):
        true_mask = np.array([[1, 1, 0], [0, 0, 1]])
        pred_mask = np.array([[1, 1, 0], [0, 1, 1]])
        res = compute_categorical_metrics(pred_mask, true_mask)

        self.assertEqual(res["POD"], 1.0)
        self.assertAlmostEqual(res["FAR"], 0.25, delta=0.01)
        self.assertAlmostEqual(res["CSI"], 0.75, delta=0.01)

    def test_low_velocity_rain_preservation(self):
        true_zh = np.array([25.0, 30.0, 35.0, 10.0])
        cleaned_zh = np.array([25.0, 30.0, -10.0, 10.0])
        vr = np.array([0.2, -0.4, 0.1, 5.0])

        ratio = compute_low_velocity_rain_preservation(cleaned_zh, true_zh, vr)
        self.assertAlmostEqual(ratio, 2.0 / 3.0, delta=0.01)

    def test_qpe_metrics(self):
        r_gauge = np.array([5.0, 10.0, 15.0, 20.0])
        r_radar = np.array([5.2, 9.8, 14.5, 20.4])
        res = compute_qpe_metrics(r_radar, r_gauge)

        self.assertTrue(res["Pearson_r"] > 0.99)
        self.assertTrue(res["NMAE_percent"] < 5.0)

if __name__ == "__main__":
    unittest.main()
