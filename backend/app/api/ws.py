from fastapi import APIRouter, WebSocket

from app.dependencies import telemetry_repo
from app.realtime.event_bus import EventBus
from app.realtime.ws_manager import WSManager

router = APIRouter(tags=["ws"])
event_bus = EventBus()
manager = WSManager(telemetry_repo)


@router.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket) -> None:
    await manager.telemetry_stream(websocket, event_bus)
