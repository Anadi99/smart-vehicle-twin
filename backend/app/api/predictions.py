from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.jwt import get_current_user
from app.dependencies import telemetry_repo
from app.ml import anomaly, drift, explainability, model_registry, predict_service

router = APIRouter(prefix="/api", tags=["predictions"])


@router.post("/predictions/run")
def run_predictions(_user: str = Depends(get_current_user)) -> dict:
    latest = telemetry_repo.latest().model_dump()
    return predict_service.run_predictions(telemetry=latest)


@router.get("/predictions/latest")
def get_latest_predictions() -> dict:
    return {"data": predict_service.latest_predictions()}


@router.get("/models/status")
def get_models_status() -> dict:
    return model_registry.registry_status()


@router.get("/predictions/explain")
def explain_predictions(_user: str = Depends(get_current_user)) -> dict:
    """SHAP / feature-importance explanation for the last RUL prediction."""
    latest = telemetry_repo.latest().model_dump()
    return explainability.explain_rul(latest)


@router.get("/anomaly/latest")
def get_anomaly() -> dict:
    """Real-time anomaly score for the latest telemetry reading."""
    latest = telemetry_repo.latest().model_dump()
    return anomaly.score(latest)


@router.get("/drift/status")
def get_drift() -> dict:
    """Data drift report — PSI per feature vs training baseline."""
    return drift.check()
