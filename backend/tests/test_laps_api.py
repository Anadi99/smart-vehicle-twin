from fastapi.testclient import TestClient

from app.main import app


def test_laps_endpoint_returns_list() -> None:
    client = TestClient(app)
    res = client.get("/api/laps")
    assert res.status_code == 200
    body = res.json()
    assert "data" in body
    assert isinstance(body["data"], list)


def test_lap_404_for_unknown_lap() -> None:
    client = TestClient(app)
    res = client.get("/api/laps/9999")
    assert res.status_code == 404
