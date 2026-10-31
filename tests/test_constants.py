import unittest
from pydwr_wg.constants import (
    RADAR_SITES,
    EARTH_RADIUS_KM,
    EFFECTIVE_EARTH_FACTOR,
    EARTH_RADIUS_4_3,
    CHANNEL_BOUNDS,
    PRECIPITATION_CLASSES,
)

class TestConstants(unittest.TestCase):
    def test_constants_definitions(self):
        self.assertEqual(EARTH_RADIUS_KM, 6371.0)
        self.assertAlmostEqual(EFFECTIVE_EARTH_FACTOR, 4.0 / 3.0, places=4)
        self.assertAlmostEqual(EARTH_RADIUS_4_3, 8494.6666, places=1)

        self.assertIn("mumbai", RADAR_SITES)
        self.assertIn("goa", RADAR_SITES)
        self.assertIn("kochi", RADAR_SITES)
        self.assertIn("mandhardev", RADAR_SITES)

        self.assertEqual(len(CHANNEL_BOUNDS), 10)
        self.assertEqual(len(PRECIPITATION_CLASSES), 5)

if __name__ == "__main__":
    unittest.main()
