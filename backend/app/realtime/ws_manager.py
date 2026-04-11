from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.repositories.telemetry_repo import TelemetryRepository
from app.realtime.event_bus import EventBus


class WSManager:
    def __init__(self, telemetry_repo: TelemetryRepository) -> None:
        self._telemetry_repo = telemetry_repo

    async def telemetry_stream(self, websocket: WebSocket, event_bus: EventBus) -> None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "snapshot",
                "data": self._telemetry_repo.latest().model_dump(),
            }
        )

        queue = event_bus.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            event_bus.unsubscribe(queue)
