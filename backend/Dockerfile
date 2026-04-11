FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml ./
COPY app ./app

RUN pip install --no-cache-dir pip setuptools wheel \
    && pip install --no-cache-dir .

EXPOSE 8000

# ── Production stage ──────────────────────────────────────────────────────
FROM base AS production

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
