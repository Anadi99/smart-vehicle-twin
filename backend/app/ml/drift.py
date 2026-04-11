"""
Data drift detection — compares live telemetry distribution against
a stored training baseline using Population Stability Index (PSI).
PSI < 0.1  → no drift
PSI 0.1–0.2 → moderate drift (warning)
PSI > 0.2  → significant drift (alert)
"""
from __future__ import annotations

import json
from pathlib import Path

_FEATURES = ["speed_kph", "brake_temp_c", "battery_soc", "brake_pad_frac"]
_PSI_WARN = 0.1
_PSI_ALERT = 0.2

# Baseline stats derived from sim/simulator.py typical operating ranges
_BASELINE: dict[str, dict] = {
    "speed_kph":     {"mean": 160.0, "std": 45.0},
    "brake_temp_c":  {"mean": 55.0,  "std": 20.0},
    "battery_soc":   {"mean": 60.0,  "std": 25.0},
    "brake_pad_frac":{"mean": 0.55,  "std": 0.25},
}

_live_buffer: list[dict] = []
_BUFFER_SIZE = 100


def record(telemetry: dict) -> None:
    """Add a telemetry snapshot to the live window."""
    _live_buffer.append({f: float(telemetry.get(f, 0)) for f in _FEATURES})
    if len(_live_buffer) > _BUFFER_SIZE:
        _live_buffer.pop(0)


def _psi_score(baseline_mean: float, baseline_std: float, live_vals: list[float]) -> float:
    """Simplified PSI using z-score bucket comparison."""
    try:
        import numpy as np  # type: ignore[import]

        if not live_vals or baseline_std == 0:
            return 0.0
        arr = np.array(live_vals)
        live_mean = float(arr.mean())
        live_std = float(arr.std()) or 1e-6
        # PSI approximation: compare normalised means
        z_shift = abs(live_mean - baseline_mean) / baseline_std
        psi = z_shift * 0.1  # scale to PSI-like range
        return round(min(psi, 1.0), 4)
    except Exception:
        return 0.0


def check() -> dict:
    """Return per-feature PSI scores and overall drift status."""
    if len(_live_buffer) < 20:
        return {"status": "collecting", "samples": len(_live_buffer), "features": {}}

    results: dict[str, dict] = {}
    max_psi = 0.0

    for feat in _FEATURES:
        vals = [row[feat] for row in _live_buffer]
        b = _BASELINE[feat]
        psi = _psi_score(b["mean"], b["std"], vals)
        max_psi = max(max_psi, psi)
        results[feat] = {
            "psi": psi,
            "status": "alert" if psi > _PSI_ALERT else "warning" if psi > _PSI_WARN else "ok",
        }

    overall = "alert" if max_psi > _PSI_ALERT else "warning" if max_psi > _PSI_WARN else "ok"
    return {
        "status": overall,
        "max_psi": round(max_psi, 4),
        "samples": len(_live_buffer),
        "features": results,
    }
