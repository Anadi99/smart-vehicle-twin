"""
Locust load test for Smart Vehicle Twin API.

Run:
    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to start the test.
"""
from __future__ import annotations

import json

from locust import HttpUser, between, task


class SVTUser(HttpUser):
    """Simulates a dashboard user polling the API."""

    wait_time = between(0.5, 2.0)
    _token: str | None = None

    def on_start(self) -> None:
        """Authenticate once per simulated user."""
        res = self.client.post(
            "/api/auth/token",
            data={"username": "admin", "password": "svtwin2026"},
        )
        if res.status_code == 200:
            self._token = res.json().get("access_token")

    def _auth_headers(self) -> dict:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    @task(5)
    def get_latest_telemetry(self) -> None:
        self.client.get("/api/telemetry/latest")

    @task(3)
    def get_telemetry_history(self) -> None:
        self.client.get("/api/telemetry/history?limit=120")

    @task(3)
    def get_alerts(self) -> None:
        self.client.get("/api/alerts/latest")

    @task(2)
    def get_laps(self) -> None:
        self.client.get("/api/laps")

    @task(2)
    def get_anomaly(self) -> None:
        self.client.get("/api/anomaly/latest")

    @task(1)
    def get_drift(self) -> None:
        self.client.get("/api/drift/status")

    @task(1)
    def run_predictions(self) -> None:
        self.client.post("/api/predictions/run", headers=self._auth_headers())

    @task(1)
    def get_health(self) -> None:
        self.client.get("/api/health")

    @task(1)
    def get_models_status(self) -> None:
        self.client.get("/api/models/status")
