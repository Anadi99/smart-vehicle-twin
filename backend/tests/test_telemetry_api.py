from fastapi.testclient import TestClient

from app.main import app


def test_get_latest_telemetry_returns_data() -> None:
    client = TestClient(app)
    response = client.get("/api/telemetry/latest")

    assert response.status_code == 200
    assert "data" in response.json()


def test_get_telemetry_history_returns_data_list() -> None:
    client = TestClient(app)
    response = client.get("/api/telemetry/history?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert "data" in body
    assert isinstance(body["data"], list)


def test_get_telemetry_history_rejects_invalid_limit() -> None:
    client = TestClient(app)
    response = client.get("/api/telemetry/history?limit=0")

    assert response.status_code == 422
