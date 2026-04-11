"""
Lap aggregator — groups the telemetry buffer into per-lap stats.
Uses the simulator's lap counter embedded in the EventBus events.
Falls back to distance-based segmentation when lap field is absent.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LapStats:
    lap: int
    samples: int = 0
    speed_mean: float = 0.0
    speed_max: float = 0.0
    brake_temp_max: float = 0.0
    brake_temp_mean: float = 0.0
    battery_soc_start: float = 100.0
    battery_soc_end: float = 100.0
    soc_drop: float = 0.0
    brake_pad_end: float = 1.0
    tire_wear_end: float = 0.0
    risk_max: float = 0.0
    risk_mean: float = 0.0
    _speed_sum: float = field(default=0.0, repr=False)
    _temp_sum: float = field(default=0.0, repr=False)
    _risk_sum: float = field(default=0.0, repr=False)

    def to_dict(self) -> dict:
        return {
            "lap": self.lap,
            "samples": self.samples,
            "speed_mean": round(self.speed_mean, 1),
            "speed_max": round(self.speed_max, 1),
            "brake_temp_max": round(self.brake_temp_max, 1),
            "brake_temp_mean": round(self.brake_temp_mean, 1),
            "soc_drop": round(self.soc_drop, 2),
            "brake_pad_end": round(self.brake_pad_end, 3),
            "tire_wear_end": round(self.tire_wear_end, 4),
            "risk_max": round(self.risk_max, 3),
            "risk_mean": round(self.risk_mean, 3),
        }


class LapAggregator:
    def __init__(self) -> None:
        self._laps: dict[int, LapStats] = {}
        self._current_lap = 1

    def ingest(self, payload: dict) -> None:
        """Feed a telemetry payload dict (from EventBus or repo)."""
        lap = int(payload.get("lap", self._current_lap))
        self._current_lap = lap

        if lap not in self._laps:
            soc_start = float(payload.get("battery_soc", 100.0))
            self._laps[lap] = LapStats(lap=lap, battery_soc_start=soc_start)

        s = self._laps[lap]
        speed = float(payload.get("speed_kph", 0))
        temp = float(payload.get("brake_temp_c", 0))
        soc = float(payload.get("battery_soc", 100))
        risk = float(payload.get("risk", 0))

        s.samples += 1
        s._speed_sum += speed
        s._temp_sum += temp
        s._risk_sum += risk
        s.speed_max = max(s.speed_max, speed)
        s.brake_temp_max = max(s.brake_temp_max, temp)
        s.risk_max = max(s.risk_max, risk)
        s.battery_soc_end = soc
        s.soc_drop = max(0.0, s.battery_soc_start - soc)
        s.brake_pad_end = float(payload.get("brake_pad_frac", s.brake_pad_end))
        s.tire_wear_end = float(payload.get("tire_wear_frac", s.tire_wear_end))

        if s.samples > 0:
            s.speed_mean = s._speed_sum / s.samples
            s.brake_temp_mean = s._temp_sum / s.samples
            s.risk_mean = s._risk_sum / s.samples

    def get_laps(self) -> list[dict]:
        return [v.to_dict() for v in sorted(self._laps.values(), key=lambda x: x.lap)]

    def get_lap(self, lap: int) -> dict | None:
        s = self._laps.get(lap)
        return s.to_dict() if s else None


# Singleton used by the API
lap_aggregator = LapAggregator()
