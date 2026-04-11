from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.lap_aggregator import lap_aggregator

router = APIRouter(prefix="/api/laps", tags=["laps"])


@router.get("")
def get_all_laps() -> dict:
    """Per-lap aggregated stats — speed, brake temp, SOC drop, risk."""
    return {"data": lap_aggregator.get_laps()}


@router.get("/{lap_number}")
def get_lap(lap_number: int) -> dict:
    lap = lap_aggregator.get_lap(lap_number)
    if lap is None:
        raise HTTPException(status_code=404, detail=f"Lap {lap_number} not found")
    return {"data": lap}
