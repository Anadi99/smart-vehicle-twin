# ADR 004 — Stateless JWT authentication for protected endpoints

**Date:** 2026-04-10  
**Status:** Accepted

## Context

The predictions/run and explain endpoints trigger ML inference — they should not be publicly callable without authentication. The app is a single-tenant demo.

## Decision

Use **stateless JWT Bearer tokens** issued by `POST /api/auth/token` (OAuth2 password flow).

## Reasons

- Stateless — no session store needed; tokens are self-contained and verified with a secret key
- Standard OAuth2 password flow — compatible with FastAPI's `OAuth2PasswordBearer` and Swagger UI's "Authorize" button
- `python-jose` handles signing/verification; `passlib` handles password hashing
- Easy to extend to multi-tenant by adding `sub` claims and a user database

## Consequences

- Tokens expire after 60 minutes (configurable via `JWT_EXPIRE_MINUTES`)
- Secret key must be rotated via `JWT_SECRET` env var in production — default is insecure
- No refresh token flow at MVP — users re-authenticate after expiry
