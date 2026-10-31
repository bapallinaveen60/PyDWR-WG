"""
Physical and operational constants for Western Ghats radar polarimetry and terrain inpainting.
"""

from typing import Dict, Any

# Geometric and Atmospheric Constants
EARTH_RADIUS_KM: float = 6371.0
EFFECTIVE_EARTH_FACTOR: float = 4.0 / 3.0
EARTH_RADIUS_4_3: float = EARTH_RADIUS_KM * EFFECTIVE_EARTH_FACTOR  # ~8494.67 km
STANDARD_REFRACTIVITY_GRADIENT: float = -39.0  # N-units/km
DIELECTRIC_FACTOR_KW2: float = 0.93  # |Kw|^2 for liquid water at meteorological radar bands
SPEED_OF_LIGHT: float = 299792458.0  # m/s

# Standard Radar Volume Dimensions
DEFAULT_N_RANGE_GATES: int = 1000
DEFAULT_RANGE_STEP_M: float = 100.0   # 100 m resolution -> 0 to 100 km
DEFAULT_N_AZIMUTHS: int = 360
DEFAULT_AZIMUTH_STEP_DEG: float = 1.0 # 1.0 deg resolution
DEFAULT_ELEVATIONS_DEG = (0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 9.0)

# Input Tensor Normalization Bounds
CHANNEL_BOUNDS = {
    "ZH": {"min": -10.0, "max": 75.0, "unit": "dBZ"},
    "ZDR": {"min": -4.0, "max": 8.0, "unit": "dB"},
    "RHOHV": {"min": 0.20, "max": 1.0, "unit": "dimensionless"},
    "PHIDP": {"min": 0.0, "max": 360.0, "unit": "deg"},
    "KDP": {"min": -1.0, "max": 10.0, "unit": "deg/km"},
    "VR": {"min": -35.0, "max": 35.0, "unit": "m/s"},
    "SW": {"min": 0.0, "max": 12.0, "unit": "m/s"},
    "DEM": {"min": 0.0, "max": 2500.0, "unit": "m MSL"},
    "BBF": {"min": 0.0, "max": 1.0, "unit": "fraction"},
    "HBEAM": {"min": 0.0, "max": 5000.0, "unit": "m AGL"},
}

CHANNEL_NAMES = [
    "ZH_norm",
    "ZDR_norm",
    "RHOHV_clipped",
    "PHIDP_norm",
    "KDP_norm",
    "VR_norm",
    "SW_norm",
    "DEM_norm",
    "BBF",
    "HBEAM_norm",
]

# Radar Station Specifications across the Western Ghats
RADAR_SITES: Dict[str, Dict[str, Any]] = {
    "mumbai": {
        "name": "Mumbai S-band DWR",
        "band": "S",
        "latitude": 18.9180,
        "longitude": 72.8250,
        "altitude_m": 60.0,
        "frequency_ghz": 2.8,
        "wavelength_m": SPEED_OF_LIGHT / (2.8 * 1e9),  # ~0.107 m
        "beamwidth_deg": 1.0,
        "blind_zone_km": 5.0,
    },
    "goa": {
        "name": "Goa S-band DWR",
        "band": "S",
        "latitude": 15.4889,
        "longitude": 73.8314,
        "altitude_m": 75.0,
        "frequency_ghz": 2.8,
        "wavelength_m": SPEED_OF_LIGHT / (2.8 * 1e9),
        "beamwidth_deg": 1.0,
        "blind_zone_km": 5.0,
    },
    "kochi": {
        "name": "Kochi S-band DWR",
        "band": "S",
        "latitude": 9.9400,
        "longitude": 76.2600,
        "altitude_m": 35.0,
        "frequency_ghz": 2.8,
        "wavelength_m": SPEED_OF_LIGHT / (2.8 * 1e9),
        "beamwidth_deg": 1.0,
        "blind_zone_km": 5.0,
    },
    "mandhardev": {
        "name": "Mandhardev X-band Radar",
        "band": "X",
        "latitude": 18.0400,
        "longitude": 73.8600,
        "altitude_m": 1300.0,
        "frequency_ghz": 9.4,
        "wavelength_m": SPEED_OF_LIGHT / (9.4 * 1e9),  # ~0.032 m
        "beamwidth_deg": 0.9,
        "blind_zone_km": 2.0,
    },
}

# In-Situ Validation Observatories
HACPL_MAHABALESHWAR = {
    "name": "High Altitude Cloud Physics Laboratory",
    "latitude": 17.9200,
    "longitude": 73.6000,
    "altitude_m": 1378.0,
    "instruments": ["JWD", "2DVD", "MRR-2"],
}

# 5-Class Categorical Precipitation Scheme for Satellite Ground-Truthing
PRECIPITATION_CLASSES = {
    0: "No-Rain",
    1: "Stratiform",
    2: "Deep Convective",
    3: "Shallow Isolated Convective",
    4: "Shallow Non-Isolated Convective",
}
