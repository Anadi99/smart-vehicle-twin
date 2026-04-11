from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

_latest: dict | None = None

# Thresholds (match configs/bmw_i4.json + alert rules)
_PAD_MIN = 0.20
_SOC_MIN = 15.0

# Lazy-loaded model cache
_models: dict[str, Any] = {}
_meta: dict = {}
_models_loaded = False


def _models_dir() -> Path:
    raw = os.environ.get("MODELS_DIR", "")
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            return p
    # walk up from this file to find the repo-root models/ dir
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "models"
        if candidate.is_dir():
            return candidate
    return Path("models")


def _load_models() -> None:
    global _models, _meta, _models_loaded
    if _models_loaded:
        return
    _models_loaded = True

    mdir = _models_dir()
    meta_path = mdir / "model_meta.json"
    if meta_path.exists():
        try:
            _meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    # Try ONNX runtime first (portable, edge-deployable)
    # Skip in test environments to avoid onnxruntime Windows teardown crash
    import os as _os
    if not _os.environ.get("PYTEST_CURRENT_TEST"):
        onnx_path = mdir / "rul_best.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort  # type: ignore[import]
                sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                _models["rul_onnx"] = sess
            except Exception:
                pass  # incompatible onnx version — joblib fallback will be used

    # joblib fallback
    try:
        import joblib  # type: ignore[import]
    except ImportError:
        return

    for key, filename in [
        ("rul", "rul_best.pkl"),
        ("soc", "soc_drop_v1.pkl"),
        ("wear", "brake_wear_v1.pkl"),
    ]:
        path = mdir / filename
        if path.exists():
            try:
                _models[key] = joblib.load(path)
            except Exception:
                pass


def _build_feature_vector(telemetry: dict) -> list[float]:
    """Build the feature vector expected by trained models."""
    _load_models()
    feature_cols: list[str] = _meta.get(
        "feature_cols",
        ["speed_mean", "speed_max", "temp_mean", "temp_max", "duration_sec"],
    )
    speed = float(telemetry.get("speed_kph", 0.0))
    temp = float(telemetry.get("brake_temp_c", 0.0))
    # For a single snapshot we use the value for both mean and max
    mapping = {
        "speed_mean": speed,
        "speed_max": speed,
        "temp_mean": temp,
        "temp_max": temp,
        "duration_sec": 60.0,
    }
    return [mapping.get(c, 0.0) for c in feature_cols]


def run_predictions(telemetry: dict | None = None) -> dict:
    """Run inference; falls back to structured nulls when models are absent."""
    _load_models()

    has_models = bool(_models)
    preds: dict[str, Any] = {
        "rul_laps": None,
        "soc_drop_per_lap": None,
        "brake_wear_per_lap": None,
        "laps_until_pad": None,
        "laps_until_soc": None,
        "recommended_action": "No prediction available",
    }

    if has_models and telemetry is not None:
        try:
            import numpy as np  # type: ignore[import]

            X = np.array([_build_feature_vector(telemetry)], dtype=float)

            # RUL — prefer ONNX, fall back to joblib
            if "rul_onnx" in _models:
                try:
                    sess = _models["rul_onnx"]
                    input_name = sess.get_inputs()[0].name
                    result = sess.run(None, {input_name: X.astype("float32")})
                    preds["rul_laps"] = round(float(result[0].reshape(-1)[0]), 2)
                    preds["rul_inference"] = "onnx"
                except Exception:
                    pass

            if preds["rul_laps"] is None and "rul" in _models:
                preds["rul_laps"] = round(float(_models["rul"].predict(X)[0]), 2)
                preds["rul_inference"] = "joblib"

            if "soc" in _models:
                preds["soc_drop_per_lap"] = round(float(_models["soc"].predict(X)[0]), 4)

            if "wear" in _models:
                preds["brake_wear_per_lap"] = round(float(_models["wear"].predict(X)[0]), 6)

            # laps-until thresholds
            curr_pad = float(telemetry.get("brake_pad_frac", 1.0))
            curr_soc = float(telemetry.get("battery_soc", 100.0))
            wear = preds["brake_wear_per_lap"]
            soc_drop = preds["soc_drop_per_lap"]

            if wear and wear > 0 and (curr_pad - _PAD_MIN) > 0:
                preds["laps_until_pad"] = int(math.floor((curr_pad - _PAD_MIN) / wear))

            if soc_drop and soc_drop > 0 and (curr_soc - _SOC_MIN) > 0:
                preds["laps_until_soc"] = int(math.floor((curr_soc - _SOC_MIN) / soc_drop))

            # recommended action
            candidates = [
                v for v in [
                    preds["laps_until_pad"],
                    preds["laps_until_soc"],
                    preds["rul_laps"],
                ]
                if v is not None
            ]
            if candidates:
                min_laps = min(candidates)
                if min_laps <= 0:
                    preds["recommended_action"] = "Pit NOW"
                elif min_laps <= 2:
                    preds["recommended_action"] = "Consider pit in 1–2 laps"
                elif min_laps <= 5:
                    preds["recommended_action"] = "Plan pit in next 3–5 laps"
                else:
                    preds["recommended_action"] = "No immediate action"

        except Exception as exc:
            preds["error"] = str(exc)

    result = {
        "predictions": preds,
        "status": "ok" if has_models else "degraded",
        "models": {
            "loaded": list(_models.keys()),
            "onnx_available": "rul_onnx" in _models,
            "meta": _meta,
        },
    }
    global _latest
    _latest = result
    return result


def latest_predictions() -> dict | None:
    return _latest
