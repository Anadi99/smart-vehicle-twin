"""
Real-time anomaly detection using Isolation Forest.
Trained on the first N telemetry samples seen at runtime (online warm-up),
then scores every subsequent reading.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

_WARMUP_SAMPLES = 200
_FEATURES = ["speed_kph", "brake_temp_c", "battery_soc", "brake_pad_frac", "tire_wear_frac", "risk"]
_CONTAMINATION = 0.05  # expected anomaly fraction

_lock = threading.Lock()
_buffer: deque[list[float]] = deque(maxlen=_WARMUP_SAMPLES)
_model: Any = None
_trained = False


def _extract(telemetry: dict) -> list[float]:
    return [float(telemetry.get(f, 0.0)) for f in _FEATURES]


def _try_train() -> None:
    global _model, _trained
    try:
        from sklearn.ensemble import IsolationForest  # type: ignore[import]
        import numpy as np  # type: ignore[import]

        X = np.array(list(_buffer))
        clf = IsolationForest(contamination=_CONTAMINATION, random_state=42, n_estimators=100)
        clf.fit(X)
        _model = clf
        _trained = True
    except Exception:
        pass


def score(telemetry: dict) -> dict:
    """
    Returns:
        is_anomaly: bool
        score: float  (more negative = more anomalous, Isolation Forest convention)
        status: 'anomaly' | 'normal' | 'warming_up'
    """
    global _trained

    feat = _extract(telemetry)

    with _lock:
        _buffer.append(feat)
        n = len(_buffer)

    if not _trained:
        if n >= _WARMUP_SAMPLES:
            _try_train()
        return {"is_anomaly": False, "score": 0.0, "status": "warming_up", "samples_collected": n}

    try:
        import numpy as np  # type: ignore[import]

        X = np.array([feat])
        raw_score = float(_model.score_samples(X)[0])
        prediction = int(_model.predict(X)[0])  # -1 = anomaly, 1 = normal
        is_anomaly = prediction == -1
        return {
            "is_anomaly": is_anomaly,
            "score": round(raw_score, 4),
            "status": "anomaly" if is_anomaly else "normal",
            "samples_collected": _WARMUP_SAMPLES,
        }
    except Exception as exc:
        return {"is_anomaly": False, "score": 0.0, "status": "error", "detail": str(exc)}


def reset() -> None:
    """Reset model — useful for testing."""
    global _model, _trained
    with _lock:
        _buffer.clear()
        _model = None
        _trained = False
