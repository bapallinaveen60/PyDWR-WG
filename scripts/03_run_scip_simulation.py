"""
Script 03: Run Semi-Synthetic Clutter Injection Pipeline (SCIP).
Superimposes dry-season clutter and virtual mountain blockage onto pristine Arabian Sea sweeps.
"""

import os
import numpy as np
from pydwr_wg.io.radar_loader import RadarDataLoader
from pydwr_wg.io.dem_loader import DEMLoader
from pydwr_wg.preprocessing.ray_tracing import compute_partial_beam_blockage
from pydwr_wg.simulation.scip_pipeline import SCIPPipeline

def main():
    print("--- Running Script 03: SCIP Clutter Injection Simulation ---")
    out_dir = "data/synthetic_scip"
    os.makedirs(out_dir, exist_ok=True)

    radar_loader = RadarDataLoader()
    dem_loader = DEMLoader(radar_lat=18.9180, radar_lon=72.8250)
    dem_polar = dem_loader.generate_western_ghats_profile()

    bbf_polar, h_beam_agl = compute_partial_beam_blockage(
        dem_polar, elevation_deg=0.5, radar_altitude_m=60.0
    )

    scip = SCIPPipeline()
    pristine_sweep = radar_loader.generate_synthetic_marine_sweep()

    x_tensor, m_clutter, zh_clean, r_clean = scip.inject_clutter_and_blockage(
        pristine_sweep=pristine_sweep,
        dem_polar=dem_polar,
        bbf_polar=bbf_polar,
        h_beam_agl=h_beam_agl,
    )

    out_file = os.path.join(out_dir, "sample_scip_pair.npz")
    np.savez_compressed(
        out_file,
        x_tensor=x_tensor,
        m_clutter=m_clutter,
        zh_clean=zh_clean,
        r_clean=r_clean,
    )

    print(f"Synthesized paired sample saved to: {out_file}")
    print(f"  Input Tensor Shape: {x_tensor.shape} (10 channels)")
    print(f"  Clutter Gate Fraction: {np.mean(m_clutter) * 100.0:.2f}%")
    print(f"  Mean Clean Reflectivity: {np.mean(zh_clean):.2f} dBZ")
    print("Script 03 completed.")

if __name__ == "__main__":
    main()
