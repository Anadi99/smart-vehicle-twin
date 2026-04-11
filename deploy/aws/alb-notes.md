# Application Load Balancer (backend)

- Target group: HTTP on container port `8000`.
- Health check path: `GET /api/health` (expect `200` and JSON `{"status":"ok"}`).
- Enable **sticky sessions** only if you add server-side session state (not required for this MVP).
- WebSocket: allow HTTP upgrade on the same listener; forward to the same target group as REST.
