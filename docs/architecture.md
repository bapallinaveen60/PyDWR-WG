# Polarimetric Terrain-Aware Inpainting Network (PTA-InNet) Architecture

## Overview
PTA-InNet is a dual-decoder encoder-decoder deep learning architecture specifically engineered to operate in radar native polar coordinates $(r, \theta)$, bypassing Cartesian regridding artifacts and preserving gate-level electromagnetic observables.

```
                     Input Polar Tensor [10 x 1000 x 360]
                                      │
                                      ▼
                        [ Stage 1: Separable Conv2d (64) ]
                                      │
                                      ▼
                        [ Stage 2: Separable Conv2d (128) ]
                                      │
                                      ▼
                        [ Stage 3: Separable Conv2d (256) ]
                                      │
                                      ▼
                        [ Stage 4: Separable Conv2d (512) ]
                                      │
                                      ▼
                        [ Stage 5: Separable Conv2d (1024) ]
                                      │
                                      ▼
                      [ Multi-Head Radial Self-Attention ]
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
     [ Semantic Segmentation Decoder ]     [ Polarimetric Inpainting Decoder ]
                   │                                     │
                   ▼                                     ▼
        Clutter Probability Mask              Clean Reflectivity ZH [dBZ]
         M_clutter in [0, 1]                   Surface Rain Rate R [mm/h]
```

## 1. Input Tensor Channels
The model ingests a 10-channel aligned polar array $\mathbf{X} \in \mathbb{R}^{10 \times 1000 \times 360}$ representing $N_r = 1000$ range gates (0 to 100 km at 100 m gate spacing) and $N_a = 360$ azimuthal radials ($1.0^\circ$ angular resolution):

1. **$Z_H$ (Horizontal Reflectivity)**: Normalized across $[-10, 75]$ dBZ.
2. **$Z_{DR}$ (Differential Reflectivity)**: Normalized across $[-4, 8]$ dB.
3. **$\rho_{HV}$ (Copolar Correlation Coefficient)**: Clipped to $[0.2, 1.0]$.
4. **$\Phi_{DP}$ (Differential Propagation Phase)**: Unwrapped and normalized across $[0, 360]^\circ$.
5. **$K_{DP}$ (Specific Differential Phase)**: Range derivative from adaptive Kalman filter $[-1, 10]^\circ/\text{km}$.
6. **$V_r$ (Radial Doppler Velocity)**: De-aliased across $[-V_{nyq}, +V_{nyq}]$.
7. **$\sigma_v$ (Doppler Spectrum Width)**: Normalized across $[0, 12]$ m/s.
8. **$H_{dem}$ (SRTM Surface Topography)**: Extracted from 30m SRTM DEM normalized across $[0, 2500]$ m MSL.
9. **$BBF$ (Partial Beam Blockage Fraction)**: Derived via 3D ray-tracing with the 4/3 Earth radius model $[0, 1]$.
10. **$h_{beam}$ (Centerline Beam Height AGL)**: Height of beam above terrain normalized across $[0, 5000]$ m.

## 2. Specialized Layers
- **Circular Azimuth Padding (`CircularAzimuthPad2d`)**: Preserves continuous spatial derivatives across the $360^\circ \to 0^\circ$ True North boundary.
- **Depthwise Separable Convolutions**: Factorizes standard 2D convolutions into spatial filtering followed by pointwise feature projection, lowering FLOPs and preventing overfitting.
- **Group Normalization (GN)**: Replaces Batch Normalization to prevent mini-batch instability caused by non-stationary meteorological scenes.
- **Attention Gates**: Modulates skip connections before concatenation, filtering out noise in unblocked oceanic sectors and focusing on complex mountain escarpments.
- **Multi-Head Radial Self-Attention**: Captures non-local correlations along radials, linking unattenuated downstream $\Phi_{DP}$ measurements with blocked near-range gates.
