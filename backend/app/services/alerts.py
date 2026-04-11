from __future__ import annotations

# Thresholds — kept in sync with configs/bmw_i4.json
_TEMP_MAX = 95.0
_TEMP_WARN = 85.0
_PAD_MIN = 0.20
_PAD_WARN = 0.30
_SOC_MIN = 15.0
_SOC_WARN = 20.0
_RISK_HIGH = 0.70
_RISK_MED = 0.40
_SPEED_MAX = 250.0


def evaluate_alerts(telemetry: dict) -> list[str]:
    """Return list of alert code strings for the given telemetry snapshot."""
    alerts: list[str] = []

    temp = float(telemetry.get("brake_temp_c", 0.0))
    pad = float(telemetry.get("brake_pad_frac", 1.0))
    soc = float(telemetry.get("battery_soc", 100.0))
    risk = float(telemetry.get("risk", 0.0))
    speed = float(telemetry.get("speed_kph", 0.0))

    # Brake temperature
    if temp > _TEMP_MAX:
        alerts.append("BRAKE_OVERHEAT")
    elif temp > _TEMP_WARN:
        alerts.append("BRAKE_TEMP_HIGH")

    # Brake pad wear
    if pad < _PAD_MIN:
        alerts.append("BRAKE_PAD_CRITICAL")
    elif pad < _PAD_WARN:
        alerts.append("BRAKE_PAD_LOW")

    # Battery SOC
    if soc < _SOC_MIN:
        alerts.append("LOW_SOC")
    elif soc < _SOC_WARN:
        alerts.append("SOC_WARNING")

    # Overall risk score
    if risk >= _RISK_HIGH:
        alerts.append("RISK_HIGH")
    elif risk >= _RISK_MED:
        alerts.append("RISK_MEDIUM")

    # Overspeed
    if speed > _SPEED_MAX:
        alerts.append("OVERSPEED")

    return alerts
