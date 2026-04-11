from app.services.alerts import evaluate_alerts


def test_low_soc_alert() -> None:
    assert "LOW_SOC" in evaluate_alerts({"battery_soc": 14.9})


def test_no_alert_when_nominal() -> None:
    alerts = evaluate_alerts({"battery_soc": 80.0, "brake_temp_c": 40.0, "brake_pad_frac": 0.5})
    assert alerts == []


def test_brake_overheat_alert() -> None:
    assert "BRAKE_OVERHEAT" in evaluate_alerts({"brake_temp_c": 100.0})


def test_brake_temp_high_warning() -> None:
    assert "BRAKE_TEMP_HIGH" in evaluate_alerts({"brake_temp_c": 90.0})


def test_brake_pad_critical() -> None:
    assert "BRAKE_PAD_CRITICAL" in evaluate_alerts({"brake_pad_frac": 0.15})


def test_brake_pad_low_warning() -> None:
    assert "BRAKE_PAD_LOW" in evaluate_alerts({"brake_pad_frac": 0.25})


def test_risk_high_alert() -> None:
    assert "RISK_HIGH" in evaluate_alerts({"risk": 0.75})


def test_risk_medium_alert() -> None:
    assert "RISK_MEDIUM" in evaluate_alerts({"risk": 0.5})


def test_overspeed_alert() -> None:
    assert "OVERSPEED" in evaluate_alerts({"speed_kph": 260.0})


def test_multiple_alerts_at_once() -> None:
    alerts = evaluate_alerts({"battery_soc": 10.0, "brake_temp_c": 110.0, "brake_pad_frac": 0.1})
    assert "LOW_SOC" in alerts
    assert "BRAKE_OVERHEAT" in alerts
    assert "BRAKE_PAD_CRITICAL" in alerts
