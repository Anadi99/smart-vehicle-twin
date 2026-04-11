from fastapi.testclient import TestClient

from app.main import app


def test_ws_telemetry_first_message_contains_type() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws/telemetry") as websocket:
        first_message = websocket.receive_json()

    assert "type" in first_message
