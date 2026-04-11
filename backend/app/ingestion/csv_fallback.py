"""
CSV fallback ingestion — polls data/latest.csv written by sim/simulator.py.
Activates automatically when MQTT is unavailable.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_LATEST_CSV = os.environ.get("LATEST_CSV", "data/latest.csv")
_POLL_INTERVAL = float(os.environ.get("CSV_POLL_INTERVAL", "1.0"))


class CsvFallback:
    def __init__(self, repo=None, event_bus=None) -> None:
        self._repo = repo
        self._event_bus = event_bus
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        """Schedule async polling loop (no-op if event loop not running)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._task = loop.create_task(self._poll_loop())
                logger.info("CSV fallback polling %s every %.1fs", _LATEST_CSV, _POLL_INTERVAL)
        except RuntimeError:
            pass  # no event loop — unit-test context

    async def _poll_loop(self) -> None:
        from app.services.normalizer import normalize_any

        last_mtime: float = 0.0
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            path = Path(_LATEST_CSV)
            if not path.exists():
                continue
            try:
                mtime = path.stat().st_mtime
                if mtime <= last_mtime:
                    continue
                last_mtime = mtime

                import csv

                with path.open(newline="") as f:
                    rows = list(csv.DictReader(f))
                if not rows:
                    continue

                row = rows[-1]
                reading = normalize_any(row)
                if self._repo:
                    self._repo.push(reading)
                if self._event_bus:
                    await self._event_bus.publish(
                        {"type": "telemetry.update", "payload": reading.model_dump()}
                    )
            except Exception as exc:
                logger.debug("CSV fallback read error: %s", exc)
