from fastapi import APIRouter

from app.dependencies import telemetry_repo
from app.services.alerts import evaluate_alerts

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/latest")
def get_latest_alerts() -> dict[str, list[str]]:
    latest = telemetry_repo.latest().model_dump()
    return {"data": evaluate_alerts(latest)}
