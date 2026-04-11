from __future__ import annotations

import os
from pathlib import Path


def _auto_models_dir() -> str | None:
    """Walk up from this file to find the repo-root models/ directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "models"
        if candidate.is_dir():
            return str(candidate)
    return None


class Settings:
    app_name: str = "Smart Vehicle Twin API"

    # Resolved in priority order: env var → auto-detect repo models/
    @property
    def models_dir(self) -> str | None:
        from_env = os.environ.get("MODELS_DIR", "").strip()
        if from_env:
            p = Path(from_env).expanduser()
            return str(p) if p.is_dir() else None
        return _auto_models_dir()

    @property
    def ingestion_mode(self) -> str:
        return os.environ.get("INGESTION_MODE", "sim").lower()

    @property
    def cors_origins(self) -> list[str]:
        raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
        return [o.strip() for o in raw.split(",")]

    @property
    def mqtt_host(self) -> str:
        return os.environ.get("MQTT_HOST", "localhost")

    @property
    def mqtt_port(self) -> int:
        return int(os.environ.get("MQTT_PORT", "1883"))

    @property
    def mqtt_topic(self) -> str:
        return os.environ.get("MQTT_TOPIC", "uvtwin/telemetry/#")


settings = Settings()
