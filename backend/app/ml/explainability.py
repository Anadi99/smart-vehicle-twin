"""
SHAP / feature-importance explainability for the RUL model.
Falls back to feature_importances_ when shap is not installed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _models_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "models"
        if candidate.is_dir():
            return candidate
    return Path("models")


def explain_rul(telemetry: dict) -> dict:
    """Return SHAP or importance-based explanation for the last RUL prediction."""
    mdir = _models_dir()
    model_path = mdir / "rul_best.pkl"
    meta_path = mdir / "model_meta.json"

    if not model_path.exists():
        return {"method": "none", "reason": "rul_best.pkl not found", "contribs": []}

    try:
        import joblib  # type: ignore[import]
        import numpy as np  # type: ignore[import]
    except ImportError:
        return {"method": "none", "reason": "joblib/numpy not installed", "contribs": []}

    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    feature_cols: list[str] = meta.get(
        "feature_cols",
        ["speed_mean", "speed_max", "temp_mean", "temp_max", "duration_sec"],
    )

    speed = float(telemetry.get("speed_kph", 0))
    temp = float(telemetry.get("brake_temp_c", 0))
    feat_vals = {
        "speed_mean": speed, "speed_max": speed,
        "temp_mean": temp,   "temp_max": temp,
        "duration_sec": 60.0,
    }
    X = np.array([[feat_vals.get(c, 0.0) for c in feature_cols]], dtype=float)

    model = joblib.load(model_path)

    # Try SHAP first
    try:
        import shap  # type: ignore[import]
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.Explainer(model.predict, X)  # type: ignore[arg-type]
        vals = explainer(X)
        shap_vals = list(map(float, vals.values.reshape(-1)))
        base = float(getattr(vals, "base_values", [0.0])[0])
        contribs = [
            {"feature": f, "value": round(feat_vals.get(f, 0.0), 4), "contribution": round(s, 6)}
            for f, s in zip(feature_cols, shap_vals)
        ]
        contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return {"method": "shap", "base_value": round(base, 4), "contribs": contribs}
    except Exception:
        pass

    # Fallback: feature_importances_
    imp: Any = getattr(model, "feature_importances_", None)
    if imp is not None:
        contribs = [
            {"feature": f, "value": round(feat_vals.get(f, 0.0), 4), "contribution": round(float(v), 6)}
            for f, v in zip(feature_cols, imp)
        ]
        contribs.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return {"method": "feature_importance", "base_value": 0.0, "contribs": contribs}

    return {"method": "none", "reason": "model has no explainability support", "contribs": []}
