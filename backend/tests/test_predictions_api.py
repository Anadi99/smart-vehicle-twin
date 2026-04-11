from fastapi.testclient import TestClient

from app.main import app


def _auth_headers(client: TestClient) -> dict:
    """Get a valid Bearer token for the demo admin user."""
    res = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "svtwin2026"},
    )
    assert res.status_code == 200, f"Login failed: {res.text}"
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_login_returns_token() -> None:
    client = TestClient(app)
    res = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "svtwin2026"},
    )
    assert res.status_code == 200
    assert "access_token" in res.json()


def test_login_rejects_bad_credentials() -> None:
    client = TestClient(app)
    res = client.post(
        "/api/auth/token",
        data={"username": "admin", "password": "wrong"},
    )
    assert res.status_code == 401


def test_predictions_run_requires_auth() -> None:
    client = TestClient(app)
    res = client.post("/api/predictions/run")
    assert res.status_code == 401


def test_predictions_run_returns_expected_keys() -> None:
    client = TestClient(app)
    headers = _auth_headers(client)
    res = client.post("/api/predictions/run", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert "predictions" in body


def test_models_status_endpoint() -> None:
    client = TestClient(app)
    res = client.get("/api/models/status")
    assert res.status_code == 200
    body = res.json()
    assert "artifact_count" in body


def test_anomaly_endpoint() -> None:
    client = TestClient(app)
    res = client.get("/api/anomaly/latest")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body


def test_drift_endpoint() -> None:
    client = TestClient(app)
    res = client.get("/api/drift/status")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body


def test_explain_requires_auth() -> None:
    client = TestClient(app)
    res = client.get("/api/predictions/explain")
    assert res.status_code == 401


def test_explain_returns_contribs_when_authed() -> None:
    client = TestClient(app)
    headers = _auth_headers(client)
    res = client.get("/api/predictions/explain", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert "method" in body
    assert "contribs" in body
