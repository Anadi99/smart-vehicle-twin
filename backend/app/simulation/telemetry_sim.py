"""
Simulated telemetry engine — mirrors sim/simulator.py logic but runs as an
asyncio background task inside the FastAPI process so the frontend always has
live data even without an external MQTT broker.
"""
from __future__ import annotations

import asyncio
import math
import random
import time


# Track shape: small oval near Nürburgring
_LAT0, _LON0 = 50.3357, 6.9470
_R_EARTH = 6_371_000.0
_RADIUS_M = 400.0

# Braking zones as lap-fraction ranges
_BRAKE_ZONES = [(0.18, 0.24), (0.52, 0.58), (0.82, 0.86)]

# Thresholds (match configs/bmw_i4.json)
_TEMP_MAX = 95.0
_PAD_MIN = 0.20
_SOC_MIN = 15.0


def _in_brake_zone(frac: float) -> bool:
    return any(a <= frac <= b for a, b in _BRAKE_ZONES)


def _gps(heading_deg: float) -> tuple[float, float]:
    rad = math.radians(heading_deg)
    dx = _RADIUS_M * math.cos(rad)
    dy = _RADIUS_M * math.sin(rad)
    dlat = (dy / _R_EARTH) * (180.0 / math.pi)
    dlon = (dx / (_R_EARTH * math.cos(math.radians(_LAT0)))) * (180.0 / math.pi)
    return round(_LAT0 + dlat, 6), round(_LON0 + dlon, 6)


def _risk_and_reasons(temp: float, pad: float, soc: float, speed: float) -> tuple[float, str]:
    reasons: list[str] = []
    risk = 0.0
    if temp > _TEMP_MAX:
        risk += min((temp - _TEMP_MAX) / 40.0, 1.0) * 0.6
        reasons.append(f"high brake temp {temp:.1f}C")
    if pad < _PAD_MIN:
        risk += min((_PAD_MIN - pad) / 0.4, 1.0) * 0.6
        reasons.append(f"low pad {pad:.2f}")
    if soc < _SOC_MIN:
        risk += min((_SOC_MIN - soc) / 20.0, 1.0) * 0.5
        reasons.append(f"low SOC {soc:.1f}%")
    if speed > 250:
        risk += 0.3
        reasons.append(f"overspeed {speed:.0f} km/h")
    return round(max(0.0, min(1.0, risk)), 3), "; ".join(reasons) if reasons else "nominal"


class TelemetrySimulator:
    """Stateful vehicle simulator — call tick() each second."""

    LAP_M = 5_100.0  # metres per lap

    def __init__(self) -> None:
        self._speed = 64.0
        self._prev_speed = 64.0
        self._brake_temp = 36.0
        self._soc = 82.5
        self._pad = 0.28
        self._tire_wear = 0.05
        self._heading = 0.0
        self._lap_dist = 0.0
        self._session_dist = 0.0
        self._lap = 1

    def tick(self, dt: float = 1.0) -> dict:
        lap_frac = self._lap_dist / self.LAP_M

        # Speed profile
        target = random.uniform(140, 240)
        if _in_brake_zone(lap_frac):
            target = random.uniform(60, 120)
        self._speed += (target - self._speed) * random.uniform(0.05, 0.15)
        self._speed = max(0.0, min(280.0, self._speed))

        accel = ((self._speed - self._prev_speed) / 3.6) / max(dt, 0.001)
        self._prev_speed = self._speed

        # Distance & GPS
        dist_m = (self._speed / 3.6) * dt
        self._lap_dist += dist_m
        self._session_dist += dist_m
        self._heading = (self._heading + (dist_m / (2 * math.pi * _RADIUS_M)) * 360.0) % 360.0
        lat, lon = _gps(self._heading)

        # Battery drain
        drain = 0.005 * dt + 0.00002 * (self._speed ** 2) * dt
        drain += max(0.0, accel) * 0.01
        self._soc = max(0.0, self._soc - drain)

        # Brake system
        if _in_brake_zone(lap_frac):
            self._brake_temp += 0.8 * (self._speed / 100.0) + random.uniform(0.5, 1.5)
            self._pad -= 0.0008 * (self._speed / 120.0)
        else:
            self._brake_temp += -(self._brake_temp - 35.0) * 0.04
        self._brake_temp = max(20.0, min(160.0, self._brake_temp))
        self._pad = max(0.05, min(1.0, self._pad))

        # Tire wear
        self._tire_wear = min(1.0, self._tire_wear + 0.0002 * (self._speed / 150.0))

        # Random failure injection (rare)
        if random.random() < 1.0 / 300.0:
            self._brake_temp += random.uniform(15, 30)
            self._soc = max(0.0, self._soc - random.uniform(1.0, 2.0))

        risk, reasons = _risk_and_reasons(self._brake_temp, self._pad, self._soc, self._speed)

        # Lap rollover
        if self._lap_dist >= self.LAP_M:
            self._lap += 1
            self._lap_dist = 0.0
            self._brake_temp = max(25.0, self._brake_temp - random.uniform(5, 10))

        return {
            "speed_kph": round(self._speed, 1),
            "brake_temp_c": round(self._brake_temp, 1),
            "battery_soc": round(self._soc, 2),
            "brake_pad_frac": round(self._pad, 3),
            "tire_wear_frac": round(self._tire_wear, 4),
            "accel_mps2": round(accel, 2),
            "lat": lat,
            "lon": lon,
            "risk": risk,
            "reasons": reasons,
            "ts": time.time(),
        }


async def run_simulator(repo, event_bus, interval: float = 1.0) -> None:
    """Background asyncio task: tick → push to repo → broadcast via EventBus."""
    from app.schemas.telemetry import Telemetry

    sim = TelemetrySimulator()
    while True:
        await asyncio.sleep(interval)
        data = sim.tick(dt=interval)
        try:
            reading = Telemetry(**{k: v for k, v in data.items() if k in Telemetry.model_fields})
            repo.push(reading)
            await event_bus.publish({"type": "telemetry.update", "payload": data})
        except Exception:
            pass
