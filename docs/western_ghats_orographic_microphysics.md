# Western Ghats Warm Orographic Precipitation Microphysics

## 1. Meteorological Context
The Western Ghats (Sahyadri) form a contiguous orographic barrier of 1,600 km extending along India's southwestern coast from Gujarat down to Kerala ($8^\circ\text{N} - 21^\circ\text{N}$). During the Indian Summer Monsoon (June–September), the low-level southwesterly monsoonal jet (LLJ) transports copious moisture from the Arabian Sea, impinging perpendicularly upon the steep coastal escarpment.

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

## 2. Microphysical Regime Differences
Precipitation physics along the windward crest differs fundamentally from continental deep convection:

| Microphysical Characteristic | Western Ghats Orographic Regimes | Continental Deep Convective Cores |
| :--- | :--- | :--- |
| **Echo Top Height** | Shallow warm tops strictly $< 4.5$ km | Deep convective towers reaching $12 - 16$ km |
| **Freezing Level ($0^\circ\text{C}$ Isotherm)** | $4.8 - 5.2$ km MSL (tops below melting layer) | $4.5 - 5.0$ km MSL (towers penetrate freezing level) |
| **Hydrometeor Growth Mechanism** | Condensation & rapid collision-coalescence | Mixed-phase, riming, graupel, and ice aggregation |
| **Drop Size Distribution ($D_m$)** | Small to medium drops ($D_m \approx 1.0 - 1.6$ mm) | Broad distribution with large drops ($D_m > 2.2$ mm) |
| **Number Concentration ($\log_{10} N_w$)** | Very high maritime concentration ($3.8 - 4.5$) | Moderate to low continental concentration ($3.0 - 3.6$) |
| **Radar Reflectivity ($Z_H$)** | Moderate ($18 - 38$ dBZ), high liquid water | High to severe ($45 - 65$ dBZ) with hail cores |
| **Doppler Kinematics** | Decelerated/tangential flow ($|V_r| < 1.0$ m/s) | Intense vertical updrafts/downdrafts ($|V_r| > 25$ m/s) |

## 3. Remote Sensing Vulnerabilities
1. **INSAT-3DR Geostationary Retrieval**:
   - Because cloud tops emit at warm thermal temperatures ($260 - 285$ K), thermal infrared thresholding algorithms (e.g. TIR1 $< 235$ K) register false negatives across the coastal escarpment.
2. **GPM DPR Spaceborne Dual-Frequency Radar**:
   - Surface clutter contamination masks the lowest $1.0 - 1.5$ km AGL.
   - Finite sensitivity floor ($\approx 12 - 18$ dBZ at Ku-band) misses light orographic drizzle.
3. **IMD Operational Ground Radars (S-band)**:
   - Low elevation angles ($0.5^\circ - 1.5^\circ$) hit rock ridges ($> 60$ dBZ clutter).
   - Operational zero-Doppler notch filters (GMAP/CLEAN-AP) mistakenly eradicate low-velocity monsoonal rain.
