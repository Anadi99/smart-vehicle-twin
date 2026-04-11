from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.alerts import router as alerts_router
from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.laps import router as laps_router
from app.api.predictions import router as predictions_router
from app.api.status import router as status_router
from app.api.telemetry import router as telemetry_router
from app.api.ws import event_bus, router as ws_router
from app.config import settings
from app.dependencies import telemetry_repo


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    ingestion_mode = settings.ingestion_mode

    if ingestion_mode == "mqtt":
        from app.ingestion.ingestion_runner import IngestionRunner
        IngestionRunner(repo=telemetry_repo, event_bus=event_bus).start()

    elif ingestion_mode == "csv":
        from app.ingestion.csv_fallback import CsvFallback
        CsvFallback(repo=telemetry_repo, event_bus=event_bus).start()

    else:
        from app.simulation.telemetry_sim import run_simulator
        asyncio.create_task(run_simulator(telemetry_repo, event_bus, interval=1.0))

    yield


app = FastAPI(title=settings.app_name, version="0.3.0", lifespan=lifespan)

# ── Prometheus metrics (/metrics) ──────────────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore[import]
    Instrumentator().instrument(app).expose(app)
except ImportError:
    pass  # optional — metrics endpoint simply won't exist

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(health_router)
app.include_router(telemetry_router)
app.include_router(alerts_router)
app.include_router(predictions_router)
app.include_router(laps_router)
app.include_router(status_router)
app.include_router(ws_router)
