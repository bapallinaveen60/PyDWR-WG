# Data Pipeline and Preprocessing Specifications

## 1. Radar Stations and Coverage
| Station | Band | Lat (°N) | Lon (°E) | Altitude (m) | Processing Radius | Primary Observational Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mumbai DWR** | S-band (2.8 GHz) | 18.9180 | 72.8250 | 60 | 100 km | Northern Western Ghats & coastal plains |
| **Goa DWR** | S-band (2.8 GHz) | 15.4889 | 73.8314 | 75 | 100 km | Central windward escarpment & Sahyadri |
| **Kochi DWR** | S-band (2.8 GHz) | 9.9400 | 76.2600 | 35 | 100 km | Southern Western Ghats & Nilgiris corridor |
| **Mandhardev** | X-band (9.4 GHz) | 18.0400 | 73.8600 | 1300 | 80 km | High-altitude complex ridge microphysics |

## 2. In-Situ Validation Anchor
- **Facility**: High Altitude Cloud Physics Laboratory (HACPL) Mahabaleshwar ($17.92^\circ\text{N}, 73.60^\circ\text{E}, 1378\text{ m MSL}$).
- **Instruments**:
  - Joss-Waldvogel Disdrometer (JWD) - 20 size bins ($0.3 - 5.4$ mm).
  - Two-Dimensional Video Disdrometer (2DVD).
  - Micro Rain Radar (MRR-2) - 24 GHz, 0 to 6 km AGL at 200 m vertical sampling.

## 3. Topographic and Satellite Datasets
- **Topography**: 30m Shuttle Radar Topography Mission (SRTM) 1-arc-second Digital Elevation Model ($14^\circ\text{N}-20^\circ\text{N}, 71^\circ\text{E}-76^\circ\text{E}$).
- **Geostationary Imagery**: INSAT-3DR/3DS Imager L1B multispectral channels (TIR1, TIR2, MIR, WV) at 4 km nadir resolution.
- **Reanalysis State**: ERA5 hourly single-level parameters (Total Column Water Vapor, Convective Available Potential Energy) at $0.25^\circ$.
