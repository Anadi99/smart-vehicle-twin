from __future__ import annotations

from pydantic import BaseModel, Field


class Telemetry(BaseModel):
    speed_kph: float = Field(..., ge=0, le=300)
    brake_temp_c: float = Field(..., ge=-20, le=1200)
    battery_soc: float = Field(..., ge=0, le=100)
    brake_pad_frac: float = Field(..., ge=0, le=1)
    tire_wear_frac: float = Field(default=0.0, ge=0, le=1)
    accel_mps2: float = Field(default=0.0)
    lat: float = Field(default=50.3357)
    lon: float = Field(default=6.9470)
    risk: float = Field(default=0.0, ge=0, le=1)
    reasons: str = Field(default="nominal")
