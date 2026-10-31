"""
CLI Utility for 3D Ray-Tracing and Beam Blockage Computation over the Western Ghats.
"""

import argparse
import os
import numpy as np
from pydwr_wg.constants import RADAR_SITES
from pydwr_wg.io.dem_loader import DEMLoader
from pydwr_wg.preprocessing.ray_tracing import compute_partial_beam_blockage

def main():
    parser = argparse.ArgumentParser(description="Compute 3D ray-tracing and beam blockage.")
    parser.add_argument("--radar", type=str, default="mumbai", choices=list(RADAR_SITES.keys()))
    parser.add_argument("--elevation", type=float, default=0.5, help="Elevation angle in degrees")
    parser.add_argument("--dem_path", type=str, default=None, help="Path to SRTM GeoTIFF")
    parser.add_argument("--output_dir", type=str, default="data/blockage_masks")
    args = parser.parse_args()

    radar_info = RADAR_SITES[args.radar]
    print(f"Executing 3D ray-tracing for {radar_info['name']} at {args.elevation} deg elevation...")

    dem_loader = DEMLoader(
        radar_lat=radar_info["latitude"],
        radar_lon=radar_info["longitude"],
        radar_alt_m=radar_info["altitude_m"],
    )
    dem_polar = dem_loader.load_dem_polar(args.dem_path)

    bbf, h_beam_agl = compute_partial_beam_blockage(
        dem_polar=dem_polar,
        elevation_deg=args.elevation,
        radar_altitude_m=radar_info["altitude_m"],
        beamwidth_deg=radar_info["beamwidth_deg"],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.radar}_el{args.elevation:.1f}_blockage.npz")
    np.savez_compressed(out_file, bbf=bbf, h_beam_agl=h_beam_agl, dem_polar=dem_polar)

    mean_bbf = np.mean(bbf)
    max_bbf = np.max(bbf)
    print(f"Ray-tracing completed successfully. Saved to: {out_file}")
    print(f"Summary: Mean BBF = {mean_bbf:.3f}, Max BBF = {max_bbf:.3f}")

if __name__ == "__main__":
    main()
