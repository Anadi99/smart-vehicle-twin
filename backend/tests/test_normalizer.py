import pytest

from app.services.normalizer import normalize_any, normalize_legacy_telemetry


def test_normalize_legacy_telemetry_maps_legacy_keys() -> None:
    payload = {
        "speed(km/h)": 72.4,
        "Temperature (°C)": 36.5,
        "Battery SOC (%)": 84.0,
        "Brake Pad (fraction)": 0.31,
    }
    normalized = normalize_legacy_telemetry(payload)
    assert normalized.speed_kph == 72.4
    assert normalized.brake_temp_c == 36.5
    assert normalized.battery_soc == 84.0
    assert normalized.brake_pad_frac == 0.31


def test_normalize_legacy_telemetry_raises_for_missing_key() -> None:
    payload = {
        "speed(km/h)": 72.4,
        "Temperature (°C)": 36.5,
        "Battery SOC (%)": 84.0,
    }
    with pytest.raises(ValueError, match="Missing required legacy telemetry keys"):
        normalize_legacy_telemetry(payload)


def test_normalize_any_accepts_canonical_keys() -> None:
    payload = {
        "speed_kph": 100.0,
        "brake_temp_c": 45.0,
        "battery_soc": 75.0,
        "brake_pad_frac": 0.5,
        "tire_wear_frac": 0.1,
        "risk": 0.2,
        "reasons": "nominal",
    }
    t = normalize_any(payload)
    assert t.speed_kph == 100.0
    assert t.tire_wear_frac == 0.1
    assert t.risk == 0.2


def test_normalize_any_accepts_pro_sim_keys() -> None:
    """Keys produced by sim/simulator.py (pro_sim) should map correctly."""
    payload = {
        "speed_kph": 155.0,
        "brake_temp_c": 88.0,
        "battery_soc": 60.0,
        "brake_pad_frac": 0.22,
        "tire_wear_frac": 0.08,
        "accel_mps2": 1.5,
        "lat": 50.3357,
        "lon": 6.9470,
        "risk": 0.35,
        "reasons": "nominal",
    }
    t = normalize_any(payload)
    assert t.lat == 50.3357
    assert t.accel_mps2 == 1.5
