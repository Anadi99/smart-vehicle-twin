from __future__ import annotations

from app.ingestion.csv_fallback import CsvFallback
from app.ingestion.mqtt_consumer import MqttConsumer


class IngestionRunner:
    def __init__(
        self,
        mqtt_consumer: MqttConsumer | None = None,
        csv_fallback: CsvFallback | None = None,
        repo=None,
        event_bus=None,
    ) -> None:
        self.mode = "mqtt"
        self._mqtt_consumer = mqtt_consumer or MqttConsumer(repo=repo, event_bus=event_bus)
        self._csv_fallback = csv_fallback or CsvFallback(repo=repo, event_bus=event_bus)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode

    def start(self) -> None:
        self._set_mode("mqtt")
        try:
            self._mqtt_consumer.start()
        except RuntimeError:
            self._set_mode("csv")
            self._csv_fallback.start()
