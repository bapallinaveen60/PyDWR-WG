"""
Script 06: Generate 1 km Cartesian 5-class precipitation ground-truth masks.
Synthesizes quality-controlled reference labels for geostationary satellite machine learning pipelines.
"""

import os
import numpy as np
from pydwr_wg.satellite.mask_generator import PrecipitationMaskGenerator
from pydwr_wg.constants import PRECIPITATION_CLASSES

def main():
    print("--- Running Script 06: Generate 5-Class Satellite Ground-Truth Labels ---")
    out_dir = "data/satellite_labels"
    os.makedirs(out_dir, exist_ok=True)

    generator = PrecipitationMaskGenerator()

    # Generate synthetic Cartesian 1 km radar reflectivity field [200, 200]
    ny, nx = 200, 200
    gx, gy = np.meshgrid(np.linspace(-100, 100, nx), np.linspace(-100, 100, ny))

    # Orographic rainband (warm shallow rain along escarpment)
    escarpment_band = (gx > 20) & (gx < 60) & (gy > -60) & (gy < 60)
    zh_grid = np.where(escarpment_band, 28.0 + np.random.normal(0, 3, (ny, nx)), 5.0)

    # Isolated convective core
    isolated_core = ((gx - 35)**2 + (gy - 20)**2) < 16
    zh_grid[isolated_core] = 42.0

    mask_5class = generator.generate_5class_mask(zh_grid)

    out_file = os.path.join(out_dir, "sample_5class_reference_mask.npz")
    np.savez_compressed(out_file, mask_5class=mask_5class, zh_1km=zh_grid)

    print(f"5-Class Reference Mask saved to: {out_file}")
    print("Class Pixel Distribution:")
    for c_id, name in PRECIPITATION_CLASSES.items():
        count = int(np.sum(mask_5class == c_id))
        pct = (count / (ny * nx)) * 100.0
        print(f"  Class {c_id} ({name}): {count} pixels ({pct:.2f}%)")

    print("Script 06 completed.")

if __name__ == "__main__":
    main()
