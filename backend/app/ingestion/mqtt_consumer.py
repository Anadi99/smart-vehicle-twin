"""
MQTT consumer — subscribes to the same topic as sim/simulator.py.
Falls back gracefully when paho-mqtt is not installed or broker is unreachable.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
_MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
_MQTT_TOPIC = os.environ.get("MQTT_TOPIC", "uvtwin/telemetry/#")


class MqttConsumer:
    def __init__(self, repo=None, event_bus=None) -> None:
        self._repo = repo
        self._event_bus = event_bus
        self._client = None

    def _on_message(self, _client, _userdata, message) -> None:  # type: ignore[no-untyped-def]
        try:
            from app.services.normalizer import normalize_any

            payload = json.loads(message.payload.decode())
            reading = normalize_any(payload)
            if self._repo:
                self._repo.push(reading)
            if self._event_bus:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        loop.create_task,
                        self._event_bus.publish(
                            {"type": "telemetry.update", "payload": reading.model_dump()}
                        ),
                    )
        except Exception as exc:
            logger.debug("MQTT message parse error: %s", exc)

    def start(self) -> None:
        try:
            import paho.mqtt.client as mqtt  # type: ignore[import]
        except ImportError:
            raise RuntimeError("paho-mqtt not installed")

        client = mqtt.Client()
        client.on_message = self._on_message
        client.connect(_MQTT_HOST, _MQTT_PORT, keepalive=30)
        client.subscribe(_MQTT_TOPIC)
        client.loop_start()
        self._client = client
        logger.info("MQTT consumer connected to %s:%s topic=%s", _MQTT_HOST, _MQTT_PORT, _MQTT_TOPIC)

    def stop(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
