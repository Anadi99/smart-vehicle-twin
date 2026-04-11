from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.ml import model_registry

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
def get_status() -> dict:
    """System status — ingestion mode, model registry, config summary."""
    reg = model_registry.registry_status()
    return {
        "ingestion_mode": settings.ingestion_mode,
        "models_dir": reg["models_dir"],
        "models_loaded": reg["artifact_count"],
        "model_artifacts": reg["artifacts"],
        "mqtt": {
            "host": settings.mqtt_host,
            "port": settings.mqtt_port,
            "topic": settings.mqtt_topic,
        },
    }
