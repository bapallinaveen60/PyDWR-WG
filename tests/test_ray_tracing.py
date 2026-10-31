import unittest
import numpy as np
from pydwr_wg.preprocessing.ray_tracing import (
    compute_beam_height,
    compute_ground_range,
    compute_partial_beam_blockage,
)

class TestRayTracing(unittest.TestCase):
    def test_beam_height(self):
        range_km = np.array([0.0, 10.0, 50.0, 100.0])
        h_beam = compute_beam_height(range_km, elevation_deg=0.5, radar_altitude_m=60.0)

        self.assertAlmostEqual(h_beam[0], 60.0, delta=1.0)
        # At 50 km and 0.5 deg, beam height should be ~400-800 m
        self.assertTrue(400.0 < h_beam[2] < 800.0)
        # Beam height must increase monotonically with range
        self.assertTrue(np.all(np.diff(h_beam) > 0))

    def test_beam_blockage_flat(self):
        nr, na = 100, 36
        flat_dem = np.zeros((nr, na), dtype=np.float32)
        bbf, h_agl = compute_partial_beam_blockage(flat_dem, elevation_deg=0.5, radar_altitude_m=60.0)

        self.assertEqual(bbf.shape, (nr, na))
        self.assertEqual(h_agl.shape, (nr, na))
        self.assertTrue(bbf[0, 0] <= 0.05)

    def test_beam_blockage_steep_mountain(self):
        nr, na = 100, 36
        mountain_dem = np.zeros((nr, na), dtype=np.float32)
        # Impose a tall ridge at gate 40 (4 km range) rising to 2000m
        mountain_dem[40:, :] = 2000.0

        bbf, h_agl = compute_partial_beam_blockage(mountain_dem, elevation_deg=0.5, radar_altitude_m=60.0)
        self.assertTrue(bbf[45, 0] > 0.90)
        for a in range(na):
            self.assertTrue(np.all(np.diff(bbf[:, a]) >= -1e-6))

if __name__ == "__main__":
    unittest.main()
