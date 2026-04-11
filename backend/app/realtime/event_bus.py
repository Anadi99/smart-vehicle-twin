import asyncio


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict]] = set()

    def subscribe(self) -> asyncio.Queue[dict]:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: dict) -> None:
        for queue in list(self._subscribers):
            await queue.put(event)
        # Feed lap aggregator from every telemetry event
        if event.get("type") == "telemetry.update":
            payload = event.get("payload", {})
            if payload:
                try:
                    from app.services.lap_aggregator import lap_aggregator
                    from app.ml import anomaly, drift
                    lap_aggregator.ingest(payload)
                    anomaly.score(payload)   # warms up the model
                    drift.record(payload)    # tracks distribution
                except Exception:
                    pass
