from fastapi import APIRouter, Query

from app.dependencies import telemetry_repo

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


@router.get("/latest")
def get_latest_telemetry() -> dict[str, dict]:
    return {"data": telemetry_repo.latest().model_dump()}


@router.get("/history")
def get_telemetry_history(limit: int = Query(default=300, ge=1, le=5000)) -> dict[str, list[dict]]:
    items = [entry.model_dump() for entry in telemetry_repo.history(limit)]
    return {"data": items}
