# ADR 002 — In-process asyncio EventBus for WebSocket fan-out

**Date:** 2026-04-10  
**Status:** Accepted

## Context

Multiple browser clients connect simultaneously via WebSocket. Telemetry arrives at 1 Hz from the simulator (or MQTT). Every connected client must receive every update with sub-250 ms latency.

## Decision

Use an **in-process asyncio `EventBus`** — a set of `asyncio.Queue` objects, one per subscriber. The ingestion layer calls `event_bus.publish()`, which puts the event into every queue. Each WebSocket handler drains its own queue.

## Reasons

- Zero external dependencies — no Redis pub/sub needed at MVP scale
- Fully async — `publish()` is a coroutine; no thread-safety issues
- Trivially testable — inject a mock EventBus in unit tests
- Scales to dozens of concurrent connections on a single Fargate task

## Consequences

- Does not scale across multiple ECS task replicas (each task has its own bus)
- Mitigation: add Redis pub/sub adapter behind the same `EventBus` interface when horizontal scaling is needed — the interface is already abstracted
