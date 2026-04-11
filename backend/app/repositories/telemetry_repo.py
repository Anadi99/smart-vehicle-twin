from __future__ import annotations

from collections import deque

from app.schemas.telemetry import Telemetry

_BUFFER_SIZE = 500

_SEED = Telemetry(
    speed_kph=64.2,
    brake_temp_c=35.8,
    battery_soc=82.5,
    brake_pad_frac=0.28,
    tire_wear_frac=0.05,
    accel_mps2=0.0,
    lat=50.3357,
    lon=6.9470,
    risk=0.0,
    reasons="nominal",
)


class TelemetryRepository:
    def __init__(self) -> None:
        self._buffer: deque[Telemetry] = deque(maxlen=_BUFFER_SIZE)
        self._buffer.append(_SEED)

    def push(self, item: Telemetry) -> None:
        self._buffer.append(item)

    def latest(self) -> Telemetry:
        return self._buffer[-1]

    def history(self, limit: int) -> list[Telemetry]:
        return list(self._buffer)[-limit:]
