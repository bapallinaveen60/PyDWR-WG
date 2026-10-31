"""
Script 01: Preprocess raw IMD radar volume scans.
Standardizes sweeps, unwraps PhiDP, and computes Kalman-filtered KDP.
"""

import sys
import numpy as np
from pydwr_wg.io.radar_loader import RadarDataLoader
from pydwr_wg.preprocessing.phase_processing import unwrap_phidp, estimate_kdp_kalman

def main():
    print("--- Running Script 01: Preprocess Raw Radars ---")
    loader = RadarDataLoader()

    # Generate or ingest sweep
    print("Ingesting / generating radar volume sweep...")
    sweep = loader.generate_synthetic_marine_sweep()

    # Preprocess PhiDP & KDP
    print("Unwrapping PhiDP and computing Kalman-filtered KDP...")
    unwrapped_phi = unwrap_phidp(sweep["PHIDP"])
    kalman_kdp = estimate_kdp_kalman(unwrapped_phi, range_step_km=0.1)

    sweep["PHIDP_unwrapped"] = unwrapped_phi
    sweep["KDP_kalman"] = kalman_kdp

    print(f"Preprocessed successfully! Sweep shape: {sweep['ZH'].shape}")
    print(f"Mean KDP: {np.mean(kalman_kdp):.4f} deg/km | Max KDP: {np.max(kalman_kdp):.4f} deg/km")

if __name__ == "__main__":
    main()
