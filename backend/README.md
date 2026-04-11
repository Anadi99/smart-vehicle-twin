# Smart Vehicle Twin backend

FastAPI service for telemetry, WebSocket streaming, alerts, and predictions.

## Local

```bash
cd backend
pip install -e ".[test]"
uvicorn app.main:app --reload --port 8000
```

## Tests

```bash
pytest
```
