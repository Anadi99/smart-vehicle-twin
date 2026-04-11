# ADR 001 — FastAPI over Django/Flask for the backend

**Date:** 2026-04-10  
**Status:** Accepted

## Context

The backend needs to serve REST endpoints, a WebSocket stream for real-time telemetry, and an ML inference pipeline — all with minimal latency and async I/O.

## Decision

Use **FastAPI** with **Uvicorn** (ASGI).

## Reasons

- Native `async/await` support — critical for the WebSocket EventBus that broadcasts to N connected clients simultaneously without blocking
- Automatic OpenAPI docs (`/docs`) — useful for portfolio demonstration and API consumer onboarding
- Pydantic v2 integration — schema validation and serialisation are zero-boilerplate
- Significantly lower overhead than Django for a stateless API service; no ORM or admin needed at MVP stage
- `TestClient` from Starlette makes unit testing endpoints trivial without spinning up a server

## Consequences

- No built-in admin panel (not needed)
- No Django ORM — persistence handled by in-memory repo + future TimescaleDB adapter
