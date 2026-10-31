# Comprehensive Eight-Month Implementation Roadmap

This project is structured across four sequential two-month phases covering data ingestion, neural architecture optimization, in-situ microphysical validation at HACPL Mahabaleshwar, and downstream INSAT-3DR satellite integration.

```
Month:     M1       M2       M3       M4       M5       M6       M7       M8
Phase:  [   Phase 1   ]   [   Phase 2   ]   [   Phase 3   ]   [   Phase 4   ]
Focus:  Ingestion & Ray   SCIP & Training  HACPL Validation  Satellite Retrain
```

---

## Phase 1: Ingestion, Coordinate Transformations, and Phase Preprocessing (Months 1–2)
### Milestone 1.1 (Month 1, Weeks 1–4)
- **Primary Actions**:
  - Ingest operational IMD S-band radar archives (Mumbai, Goa, Kochi) and Mandhardev X-band research radar.
  - Standardize raw Rainbow, IRIS, and NetCDF formats into unified polar tensors ($1000 \times 360$).
  - Execute 3D ray-tracing simulations over 30m SRTM topography to compute line-of-sight Partial Beam Blockage Fraction (BBF) and beam centerline height $h_{beam}$.
- **Verification Deliverables**:
  - Universal polar radar pipeline.
  - Gate-level BBF and beam height matrices for tilts $0.5^\circ - 9.0^\circ$.

### Milestone 1.2 (Month 2, Weeks 5–8)
- **Primary Actions**:
  - Implement $\Phi_{DP}$ transmitter offset removal and $180^\circ$ phase unwrapping.
  - Deploy adaptive discrete Kalman filter to extract range-resolved $K_{DP}$ profiles while filtering $\delta_{CO}$ anomalies.
  - Composite 150 pre-monsoon dry scans (March–May) to construct static clutter probability masks.
- **Verification Deliverables**:
  - Robust differential phase processing engine.
  - Baseline clear-air clutter probability masks and main-bang exclusion zones.

---

## Phase 2: Network Engineering, Clutter Simulation, and Model Training (Months 3–4)
### Milestone 2.1 (Month 3, Weeks 9–12)
- **Primary Actions**:
  - Build Semi-Synthetic Clutter Injection Pipeline (SCIP) using pristine Arabian Sea marine rain sweeps.
  - Superimpose dry-season clutter profiles and simulated mountain blockage while preserving downstream $\Phi_{DP}$.
  - Compile the 10-channel dual-decoder Attention U-Net (`PTAInNet`).
- **Verification Deliverables**:
  - Curated paired training dataset ($> 25,000$ polar sweeps).
  - Production-ready PyTorch model with circular azimuth padding.

### Milestone 2.2 (Month 4, Weeks 13–16)
- **Primary Actions**:
  - Execute multi-GPU training on high-performance computing clusters (NVIDIA A100 GPUs).
  - Run ablation studies on loss components ($\lambda_1 = 1.0, \lambda_2 = 2.5, \lambda_3 = 0.8, \lambda_4 = 1.5$).
  - Optimize hyperparameters using AdamW with Cosine Annealing learning rate schedule.
- **Verification Deliverables**:
  - Fully converged model checkpoint with validated composite loss.

---

## Phase 3: In-Situ Microphysical Validation and Escarpment Benchmarking (Months 5–6)
### Milestone 3.1 (Month 5, Weeks 17–20)
- **Primary Actions**:
  - Collocate predictions with High Altitude Cloud Physics Laboratory (HACPL) Mahabaleshwar ($17.92^\circ\text{N}, 73.60^\circ\text{E}, 1378\text{ m MSL}$).
  - Ingest 1-minute drop size distribution (DSD) data from JWD and 2DVD.
  - Evaluate lowest 1.5 km AGL boundary-layer reflectivity profiles against collocated 24 GHz Micro Rain Radar (MRR-2).
- **Verification Deliverables**:
  - Boundary-layer reflectivity bias constrained within $[-1.0, +1.0]$ dB (vs $-6.5$ to $-8.5$ dB operational baseline).

### Milestone 3.2 (Month 6, Weeks 21–24)
- **Primary Actions**:
  - Derive warm-rain polarimetric QPE estimators ($R(K_{DP}), R(K_{DP}, Z_{DR})$).
  - Benchmark PTA-InNet against operational GMAP notch filters, Gabella texture filter, and Fuzzy Logic CMD.
  - Verify quantitative rainfall estimation across IMD Automatic Weather Station (AWS) networks.
- **Verification Deliverables**:
  - QPE NMAE $\le 18.0\%$ and low-velocity rain preservation $\ge 0.94$.

---

## Phase 4: Satellite Label Synthesis, Model Retraining, and Manuscript Preparation (Months 7–8)
### Milestone 4.1 (Month 7, Weeks 25–28)
- **Primary Actions**:
  - Process 5 years of monsoon volume scans (JJAS 2020–2024) across Mumbai, Goa, and Kochi radars.
  - Project de-cluttered fields onto a 1 km $\times$ 1 km Cartesian grid and synthesize 5-class precipitation masks.
  - Collocate reference masks with INSAT-3DR Imager multispectral channels and ERA5 thermodynamic fields.
- **Verification Deliverables**:
  - 5-year quality-controlled ground-truth catalogue.
  - Retrained 8-channel hierarchical U-Net satellite precipitation model.

### Milestone 4.2 (Month 8, Weeks 29–32)
- **Primary Actions**:
  - Assess satellite shallow warm-rain recall improvements (target: $\ge 0.62$ vs $0.40 - 0.46$ baseline).
  - Package open-source library (`PyDWR-WG`).
  - Prepare and submit research manuscript to *IEEE Transactions on Geoscience and Remote Sensing (TGRS)*.
- **Verification Deliverables**:
  - Public GitHub repository release.
  - Complete research manuscript ready for IEEE submission.
