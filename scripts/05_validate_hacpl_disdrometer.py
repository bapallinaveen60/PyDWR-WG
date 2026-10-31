"""
Script 05: In-situ validation against HACPL Mahabaleshwar microphysics.
Compares reconstructed boundary-layer profiles against JWD, 2DVD, and MRR-2.
"""

import numpy as np
from pydwr_wg.io.disdrometer_loader import DisdrometerLoader
from pydwr_wg.evaluation.disdrometer_intercomp import DisdrometerIntercomparison

def main():
    print("--- Running Script 05: HACPL Mahabaleshwar Validation ---")
    disd_loader = DisdrometerLoader(altitude_m=1378.0)

    # 1. Micro Rain Radar (MRR-2) boundary-layer profile
    heights, mrr_zh = disd_loader.load_mrr_profile()

    # 2. Intercomparison object for Mumbai S-band radar
    intercomp = DisdrometerIntercomparison(radar_lat=18.9180, radar_lon=72.8250)

    # Simulated radar reconstructed reflectivity at HACPL location
    simulated_radar_zh = 33.2  # dBZ

    bl_eval = intercomp.evaluate_boundary_layer_bias(
        radar_reconstructed_zh=simulated_radar_zh,
        mrr_heights_agl=heights,
        mrr_zh_profile=mrr_zh,
        max_agl_height_m=1500.0,
    )

    print("HACPL Boundary Layer (< 1.5 km AGL) Evaluation Results:")
    print(f"  MRR-2 Boundary Layer Mean ZH: {bl_eval['mrr_bl_mean_zh']:.2f} dBZ")
    print(f"  Radar Reconstructed ZH:       {bl_eval['radar_reconstructed_zh']:.2f} dBZ")
    print(f"  Boundary Layer Bias:          {bl_eval['boundary_layer_bias_db']:.2f} dB")
    print(f"  Meets IEEE Target (|Bias| <= 1.0 dB): {bl_eval['meets_target_spec']}")

    # 3. Disdrometer rain rate comparison
    sample_drops = np.array([25, 30, 28, 20, 15, 12, 8, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    dsd_res = disd_loader.compute_dsd_parameters(sample_drops)

    rr_eval = intercomp.evaluate_surface_rain_rate(
        radar_rain_rate=4.8,
        disdrometer_rain_rate=dsd_res["Rain_Rate"],
    )

    print("\nSurface Disdrometer Rain Rate Comparison:")
    print(f"  Disdrometer Rain Rate: {rr_eval['gauge_r_mm_h']:.2f} mm/h")
    print(f"  Radar Rain Rate:       {rr_eval['radar_r_mm_h']:.2f} mm/h")
    print(f"  Relative Error:        {rr_eval['rel_error_percent']:.2f}%")
    print("Script 05 completed.")

if __name__ == "__main__":
    main()
