from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.schemas.telemetry import Telemetry

# Legacy CSV / old-simulator key → canonical field
LEGACY_KEY_MAP: dict[str, str] = {
    "speed(km/h)": "speed_kph",
    "Temperature (°C)": "brake_temp_c",
    "Battery SOC (%)": "battery_soc",
    "Brake Pad (fraction)": "brake_pad_frac",
    # pro_sim / sim/simulator.py keys (already canonical, kept for safety)
    "brake_temp_c": "brake_temp_c",
    "battery_soc": "battery_soc",
    "brake_pad_frac": "brake_pad_frac",
    "speed_kph": "speed_kph",
    # extra fields from pro_sim
    "tire_wear_frac": "tire_wear_frac",
    "accel_mps2": "accel_mps2",
    "lat": "lat",
    "lon": "lon",
    "risk": "risk",
    "reasons": "reasons",
}

_REQUIRED_LEGACY = {"speed(km/h)", "Temperature (°C)", "Battery SOC (%)", "Brake Pad (fraction)"}


def normalize_legacy_telemetry(payload: Mapping[str, Any]) -> Telemetry:
    """Map old CSV column names to canonical Telemetry schema."""
    missing = [k for k in _REQUIRED_LEGACY if k not in payload]
    if missing:
        raise ValueError(f"Missing required legacy telemetry keys: {', '.join(missing)}")

    mapped: dict[str, Any] = {}
    for src, dst in LEGACY_KEY_MAP.items():
        if src in payload:
            mapped[dst] = payload[src]

    return Telemetry(**{k: v for k, v in mapped.items() if k in Telemetry.model_fields})


def normalize_any(payload: Mapping[str, Any]) -> Telemetry:
    """Accept both legacy and canonical payloads — used by MQTT + CSV ingestion."""
    mapped: dict[str, Any] = {}
    for src, dst in LEGACY_KEY_MAP.items():
        if src in payload:
            mapped[dst] = payload[src]
    # canonical fields override
    for field in Telemetry.model_fields:
        if field in payload:
            mapped[field] = payload[field]
    return Telemetry(**{k: v for k, v in mapped.items() if k in Telemetry.model_fields})
