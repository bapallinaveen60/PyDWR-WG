"""
Script 02: Compute DEM-based 3D ray-tracing beam blockage.
Computes Partial Beam Blockage Fraction (BBF) and centerline height AGL for Western Ghats radars.
"""

import os
import numpy as np
from pydwr_wg.constants import RADAR_SITES
from pydwr_wg.io.dem_loader import DEMLoader
from pydwr_wg.preprocessing.ray_tracing import compute_partial_beam_blockage

def main():
    print("--- Running Script 02: 3D Ray-Tracing Topographic Blockage ---")
    out_dir = "data/blockage_masks"
    os.makedirs(out_dir, exist_ok=True)

    for site_key in ["mumbai", "goa"]:
        radar = RADAR_SITES[site_key]
        print(f"Processing site: {radar['name']} ({site_key})...")

        dem_loader = DEMLoader(
            radar_lat=radar["latitude"],
            radar_lon=radar["longitude"],
            radar_alt_m=radar["altitude_m"],
        )
        dem_polar = dem_loader.generate_western_ghats_profile()

        # Compute for 0.5 deg (lowest tilt, critical for warm shallow rain)
        bbf_05, h_beam_05 = compute_partial_beam_blockage(
            dem_polar, elevation_deg=0.5, radar_altitude_m=radar["altitude_m"]
        )

        out_path = os.path.join(out_dir, f"{site_key}_el0.5_blockage.npz")
        np.savez_compressed(out_path, bbf=bbf_05, h_beam_agl=h_beam_05, dem_polar=dem_polar)
        print(f"  Saved blockage mask to {out_path} (Mean BBF: {np.mean(bbf_05):.3f})")

    print("Script 02 completed.")

if __name__ == "__main__":
    main()
