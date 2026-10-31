# PyDWR-WG: Ground Polarimetric Doppler Radar De-Cluttering and Low-Level Warm Orographic Precipitation Reconstruction Over the Western Ghats

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Domain: Western Ghats](https://img.shields.io/badge/Domain-Western%20Ghats%2C%20India-success.svg)]()
[![Target: IEEE TGRS](https://img.shields.io/badge/Publication-IEEE%20TGRS-blueviolet.svg)]()

> **An Operational Deep-Learning Framework for High-Fidelity Satellite Ground-Truthing under Complex Tropical Topography**

---

## 🌧️ Physical Context and Observational Vulnerabilities

The Western Ghats of peninsular India intercept the low-level southwesterly monsoonal jet traversing the Arabian Sea during the summer monsoon (June–September). The steep escarpment, ascending from coastal plains to altitudes exceeding 1,500 to 2,000 m MSL within 40 to 60 km, triggers intense moist orographic ascent and produces seasonal accumulations exceeding 5,000 mm.

Unlike continental deep convection (where towers reach 12–16 km with pronounced mixed-phase microphysics), Western Ghats orographic clouds are **predominantly shallow and warm-topped (< 4.5 km altitude)**, situated well below the environmental melting layer ($0^\circ\text{C}$ isotherm at 4.8–5.2 km MSL). Precipitation generation is governed almost exclusively by collision-coalescence drop growth in high-liquid-water boundary-layer clouds.

```
Elevation (m MSL)
 2000 |                              [Western Ghats Crest]
      |                                  / \  (HACPL Mahabaleshwar 1378m)
 1500 |                                 /   \
      |      [Shallow Warm Clouds]     /     \     [Deccan Plateau]
 1000 |         (Tops < 4.5 km)       /       \__________________
      |       ~~~~~~~~~~~~~~~~~~~~~  /
  500 |      ~~~~~~~~~~~~~~~~~~~~~~~
      |  [Arabian Sea]    [Coastal Plains]
    0 |───────────────────────40-60 km───────
```

### The Compounding Error Chain in Existing Systems:
1. **Satellite Infrared Failure**: Cloud-top brightness temperatures in the thermal window (TIR1: 10.8 µm) are warm (260–285 K), mimicking non-raining marine stratocumulus and causing conventional cloud-top threshold algorithms to register severe false negatives.
2. **Spaceborne Radar (GPM DPR) Constraints**: Heavy ground clutter masks the lowest 1.0–1.5 km AGL, while the finite sensitivity floor (~12–18 dBZ) misses light warm drizzle (< 1 mm/h).
3. **Ground Radar Clutter & Blockage**: Operational IMD S-band radars (Mumbai, Goa, Kochi) suffer from main-bang saturation (0–5 km), severe terrain ground clutter (> 60 dBZ), and partial/total beam blockage (PBB/TBB).
4. **Failure of Operational Doppler Notch Filters**: Monsoonal flow frequently stagnates or turns parallel to ridge axes relative to coastal radars ($|V_r| < 1.0$ m/s). Operational zero-Doppler notch filters (GMAP, CLEAN-AP) mistakenly eradicate authentic orographic rain as ground clutter.

---

## ⚡ The Solution: PTA-InNet Architecture

**Polarimetric Terrain-Aware Inpainting Network (PTA-InNet)** operates directly in the radar native polar coordinate system $(r, \theta)$ across 10 aligned channels:

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

### 10 Aligned Input Channels:
1. $Z_H$ (Horizontal Reflectivity): $[-10, 75]$ dBZ
2. $Z_{DR}$ (Differential Reflectivity): $[-4, 8]$ dB
3. $\rho_{HV}$ (Copolar Correlation Coefficient): $[0.2, 1.0]$
4. $\Phi_{DP}$ (Differential Propagation Phase): Unwrapped $[0, 360]^\circ$
5. $K_{DP}$ (Specific Differential Phase): Kalman-filtered range derivative
6. $V_r$ (Radial Doppler Velocity): De-aliased across Nyquist interval
7. $\sigma_v$ (Doppler Spectrum Width): $[0, 12]$ m/s
8. $H_{dem}$ (Surface Topography): 30m SRTM DEM altitude [m MSL]
9. $BBF$ (Partial Beam Blockage Fraction): 3D ray-tracing 4/3 Earth radius model $[0, 1]$
10. $h_{beam}$ (Centerline Beam Height AGL): Height above terrain [m]

---

## 🔬 Scientific & Theoretical Innovations

1. **Phase-Preserving Physical Inpainting ($\mathcal{L}_{\text{phys}}$)**: While reflectivity ($Z_H$) is attenuated by mountain blockage, differential propagation phase ($\Phi_{DP}$) is invariant to power blockage:
   $$\left(\frac{\partial \Phi_{DP}}{\partial r}\right)_{\text{blocked}} = \left(\frac{\partial \Phi_{DP}}{\partial r}\right)_{\text{unblocked}} = 2 K_{DP}$$
   PTA-InNet enforces that the path integral of reconstructed specific differential phase matches the observed downstream phase accumulation, preventing unphysical hallucinations.
2. **Disentanglement of Low-Velocity Monsoonal Warm Rain**: Separates non-meteorological targets from genuine hydrometeors in multi-dimensional polarimetric space ($\rho_{HV}, Z_{DR}, K_{DP}$) without zero-Doppler velocity filtering, preserving stagnated monsoonal rain ($|V_r| < 1.0$ m/s).
3. **Semi-Synthetic Clutter Injection Pipeline (SCIP)**: Solves ground-truth scarcity by superimposing dry-season clutter profiles and simulated terrain blockage onto pristine Arabian Sea marine rain sweeps while preserving downstream phase dynamics.
4. **Resolution of Satellite Ground-Truth Label Contamination**: Delivers a quality-controlled 1 km Cartesian 5-class reference mask to retrain geostationary satellite machine learning models (INSAT-3DR/3DS).

---

## 📊 Verification Standards and Benchmarking

| Evaluation Domain | Verification Metric | Operational Baseline (GMAP / Thresholding) | Proposed PTA-InNet Target | Scientific Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Clutter Discrimination** | Critical Success Index (CSI) | 0.68–0.74 (Operational CMD) | **$\ge 0.88$** | Minimizes misclassification of rock targets |
| **False Rain Alarms** | False Alarm Rate (FAR) | 0.18–0.24 (Residual clutter) | **$\le 0.05$** | Eliminates spurious light rain from terrain |
| **Low-Velocity Rain** | Echo Preservation ($|V_r| < 1$ m/s) | 0.45–0.52 (Notch filter loss) | **$\ge 0.94$** | Rescues stagnated orographic rain |
| **Sector Inpainting** | Structural Similarity (SSIM) | 0.58–0.64 (Standard interpolation) | **$\ge 0.86$** | Reconstructs realistic rainbands behind ridges |
| **Reconstruction Fidelity**| Peak Signal-to-Noise Ratio (PSNR) | 18.2–21.5 dB (Gaussian filling) | **$\ge 28.5$ dB** | Restores continuous reflectivity fields |
| **Rain Gauge QPE** | Normalized Mean Absolute Error | 38.5%–46.0% (Reflectivity Z–R) | **$\le 18.0\%$** | Blockage-invariant phase constraints |
| **Microphysical Profile** | Boundary Layer Bias (< 1.5 km) | -6.5 to -8.5 dB (Overshooting) | **$[-1.0, +1.0]$ dB** | Matches HACPL MRR-2 vertical profiles |
| **Satellite Retrieval** | Western Ghats Shallow-Rain Recall | 0.40–0.46 (Trained on GPM DPR) | **$\ge 0.62$** | Resolves INSAT under-detection of warm rain |

---

## 📁 Repository Structure

```
├── .github/workflows/ci.yml       # Automated CI workflow (PyTorch testing)
├── .gitignore                     # Git ignore rules for radar binaries & DEMs
├── LICENSE                        # MIT License
├── README.md                      # Comprehensive project documentation
├── pyproject.toml                 # Package specifications & dependencies
├── setup.py                       # Editable pip installation
├── requirements.txt               # Direct pip requirements
├── environment.yml                # Conda / Mamba virtual environment
├── configs/
│   ├── config.yaml                # Master configuration
│   ├── model/pta_innet.yaml       # PTA-InNet architecture parameters
│   ├── data/western_ghats_radars.yaml # Radar coordinates & specifications
│   ├── training/train_scip.yaml   # SCIP & GPU training hyperparameters
│   └── satellite/insat3dr_collocation.yaml # INSAT-3DR grid & classes
├── data/
│   ├── README.md                  # Layout guide for radar & DEM archives
│   ├── raw/                       # Raw IMD, Mandhardev, SRTM, HACPL, INSAT
│   ├── processed/                 # Calibrated polar arrays [1000, 360]
│   ├── clutter_maps/              # Dry-weather composite clutter probability
│   ├── blockage_masks/            # 3D ray-tracing BBF and beam height arrays
│   ├── synthetic_scip/            # SCIP paired training datasets
│   └── satellite_labels/          # 1 km gridded 5-class reference masks
├── docs/
│   ├── architecture.md            # In-depth PTA-InNet neural architecture
│   ├── mathematical_formulation.md# Scattering physics & composite loss
│   ├── western_ghats_orographic_microphysics.md # Microphysical regimes
│   ├── eight_month_roadmap.md     # 4-phase, 8-month milestone plan
│   └── data_pipeline.md           # Dataset descriptions & coordinates
├── pydwr_wg/                      # Core Python library
│   ├── constants.py               # Radar sites, physics, dielectric factor
│   ├── io/                        # Radar, DEM, INSAT, and disdrometer loaders
│   ├── preprocessing/             # 3D ray-tracing, phase unwrapping, Kalman KDP
│   ├── simulation/                # SCIP clutter injection pipeline
│   ├── models/                    # Circular layers, Attention, PTAInNet, Losses
│   ├── physics/                   # Self-consistency, QPE estimators, DSD
│   ├── evaluation/                # CSI, SSIM, PSNR, NMAE, baseline benchmarks
│   ├── satellite/                 # INSAT-3DR collocation & 5-class masks
│   └── pipelines/                 # Training, inference, ray-tracing CLI
├── scripts/
│   ├── 01_preprocess_raw_radars.py     # Unwraps PhiDP & computes Kalman KDP
│   ├── 02_compute_dem_blockage.py      # Computes 3D ray-tracing BBF masks
│   ├── 03_run_scip_simulation.py       # Generates SCIP paired training sweeps
│   ├── 04_train_pta_innet.py           # Trains PTA-InNet with composite loss
│   ├── 05_validate_hacpl_disdrometer.py# In-situ HACPL Mahabaleshwar validation
│   ├── 06_generate_satellite_labels.py # Synthesizes 1 km 5-class masks
│   └── 07_evaluate_insat_retrieval.py  # Evaluates satellite shallow rain recall
└── tests/
    ├── test_constants.py
    ├── test_ray_tracing.py
    ├── test_phase_processing.py
    ├── test_layers_and_model.py
    ├── test_losses.py
    ├── test_qpe_and_physics.py
    └── test_metrics.py
```

---

## 🚀 Installation

### Option 1: Conda / Mamba Environment (Recommended)
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd "Ground Polarimetric Doppler Radar De-Cluttering and Low-Level Warm Orographic Precipitation Reconstruction Over the Western Ghats"
conda env create -f environment.yml
conda activate pydwr-wg
pip install -e .
```

### Option 2: Pip Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## 🧪 Running Unit Tests
The test suite validates polar layers, ray-tracing, Kalman filtering, losses, and metrics:
```bash
python -m unittest discover -s tests -v
```

---

## 🏃 Step-by-Step Execution Pipeline

1. **Preprocess Raw Radars and Compute Kalman $K_{DP}$**:
   ```bash
   python scripts/01_preprocess_raw_radars.py
   ```
2. **Compute 3D Ray-Tracing Topographic Blockage (BBF)**:
   ```bash
   python scripts/02_compute_dem_blockage.py
   ```
3. **Generate Paired Semi-Synthetic Training Data (SCIP)**:
   ```bash
   python scripts/03_run_scip_simulation.py
   ```
4. **Train PTA-InNet Model**:
   ```bash
   python scripts/04_train_pta_innet.py --epochs 25 --batch_size 4
   ```
5. **Validate Against HACPL Mahabaleshwar In-Situ Disdrometer & MRR-2**:
   ```bash
   python scripts/05_validate_hacpl_disdrometer.py
   ```
6. **Synthesize 1 km Cartesian 5-Class Satellite Ground-Truth Labels**:
   ```bash
   python scripts/06_generate_satellite_labels.py
   ```
7. **Evaluate INSAT-3DR Shallow Warm-Rain Retrieval**:
   ```bash
   python scripts/07_evaluate_insat_retrieval.py
   ```

---

## 💻 Python API Usage Example

```python
import torch
from pydwr_wg.pipelines.inference import PTAInferencePipeline
from pydwr_wg.physics.qpe_estimators import adaptive_polarimetric_qpe

# 1. Initialize production inference engine
pipeline = PTAInferencePipeline(device="cuda" if torch.cuda.is_available() else "cpu")

# 2. Ingest 10-channel normalized polar radar tensor [10, Nr=1000, Na=360]
# (ZH, ZDR, RHOHV, PHIDP, KDP, VR, SW, DEM, BBF, HBEAM)
x_tensor = ... 

# 3. Predict de-cluttered fields
results = pipeline.process_polar_tensor(x_tensor)
clutter_mask = results["clutter_mask"]   # [1000, 360]
clean_zh = results["zh_clean"]           # [1000, 360] dBZ
rain_rate = results["rain_rate"]         # [1000, 360] mm/h
```

---

## 🔗 How to Connect to Your GitHub Repository

When you create your remote GitHub repository, connect and push this local repository using the following terminal commands:

```bash
# 1. Add your GitHub remote URL
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git

# 2. Ensure main branch is active
git branch -M main

# 3. Stage and commit all files
git add .
git commit -m "Initial commit: PyDWR-WG operational radar decluttering and warm orographic precipitation reconstruction framework"

# 4. Push to GitHub
git push -u origin main
```

---

## 📜 Citation

If you use this framework in your research, please cite:

```bibtex
@article{geovision2026pydwr,
  title={Ground Polarimetric Doppler Radar De-Cluttering and Low-Level Warm Orographic Precipitation Reconstruction Over the Western Ghats: An Operational Framework for High-Fidelity Satellite Ground-Truthing},
  author={GeoVision Research Group and Collaborators},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  note={Under Review}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
