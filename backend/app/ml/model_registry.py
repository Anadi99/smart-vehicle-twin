from __future__ import annotations

from pathlib import Path

from app.config import settings


def _resolve_models_dir() -> Path | None:
    raw = settings.models_dir
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_dir() else None
    return None


def list_model_artifacts() -> list[dict[str, str]]:
    """Return discovered model files (pkl/onnx/joblib) under the models directory."""
    root = _resolve_models_dir()
    if root is None:
        return []
    out: list[dict[str, str]] = []
    for pattern in ("*.pkl", "*.onnx", "*.joblib"):
        for path in sorted(root.glob(pattern)):
            out.append({
                "name": path.name,
                "path": str(path.resolve()),
                "kind": path.suffix.lower(),
            })
    return out


def registry_status() -> dict[str, object]:
    artifacts = list_model_artifacts()
    models_dir = _resolve_models_dir()
    return {
        "models_dir": str(models_dir) if models_dir else None,
        "models_dir_configured": models_dir is not None,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
