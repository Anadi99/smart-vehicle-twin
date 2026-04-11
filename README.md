# Smart Vehicle Twin

[![CI](https://github.com/your-org/smart-vehicle-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/smart-vehicle-twin/actions/workflows/ci.yml)
[![Deploy](https://github.com/your-org/smart-vehicle-twin/actions/workflows/deploy.yml/badge.svg)](https://github.com/your-org/smart-vehicle-twin/actions/workflows/deploy.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![AWS](https://img.shields.io/badge/AWS-ECS%20%7C%20S3%20%7C%20CloudFront-FF9900?logo=amazonaws)
![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC?logo=terraform)

A **cloud-native AI-powered digital twin** for a BMW i4 race car.  
Real-time telemetry streams from a physics-based simulator (or live MQTT broker) through a FastAPI backend, ML inference pipeline, and a React dashboard — all deployable to AWS with one Terraform apply.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          AWS Cloud                                  │
│                                                                     │
│  ┌──────────────┐   HTTPS    ┌──────────────────────────────────┐  │
│  │  CloudFront  │◄──────────►│  S3 (React SPA static assets)   │  │
│  │  + WAF       │            └──────────────────────────────────┘  │
│  └──────┬───────┘                                                   │
│         │ /api/*  /ws/*                                             │
│  ┌──────▼───────┐            ┌──────────────────────────────────┐  │
│  │     ALB      │◄──────────►│  ECS Fargate (FastAPI + uvicorn) │  │
│  └──────────────┘            │  ┌────────────┐ ┌─────────────┐ │  │
│                              │  │ Sim loop   │ │ ML models   │ │  │
│                              │  │ (asyncio)  │ │ (joblib)    │ │  │
│                              │  └────────────┘ └─────────────┘ │  │
│                              └──────────────────────────────────┘  │
│                                        ▲                            │
│                              ┌─────────┴────────┐                  │
│                              │  ECR (Docker img) │                  │
└──────────────────────────────┴──────────────────┴──────────────────┘

Local / IoT edge
┌──────────────────────────────────────────────────────┐
│  sim/simulator.py  ──MQTT──►  Mosquitto broker       │
│  (BMW i4 physics)            (docker-compose.mqtt.yml)│
└──────────────────────────────────────────────────────┘
```

### Data flow

```
Simulator tick (1 Hz)
  → MQTT publish  (or CSV write as fallback)
    → FastAPI ingestion layer
      → TelemetryRepository (rolling 500-sample buffer)
        → EventBus.publish()
          → WebSocket broadcast to all connected browsers
          → Alert engine evaluation
          → ML inference (RUL / SOC drop / brake wear)
            → Predictions API
              → React dashboard update
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Simulator | Python, paho-mqtt, NumPy |
| Backend | FastAPI, Uvicorn, Pydantic v2, joblib, scikit-learn |
| ML models | RandomForest / XGBoost (RUL, SOC drop, brake wear) |
| Frontend | React 19, Vite, TypeScript, Tailwind CSS v4, Recharts, Three.js |
| Realtime | WebSocket (FastAPI native), asyncio EventBus |
| Ingestion | MQTT primary → CSV fallback → built-in sim |
| Containers | Docker, docker-compose |
| Cloud | AWS ECS Fargate, ALB, ECR, S3, CloudFront |
| IaC | Terraform >= 1.6 |
| CI/CD | GitHub Actions |

---

## Quick start (local dev)

**Backend** (terminal 1):
```bash
cd backend
pip install -e ".[test]"
uvicorn app.main:app --reload --port 8000
# INGESTION_MODE=sim by default — built-in simulator starts automatically
```

**Frontend** (terminal 2):
```bash
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

**Full stack via Docker:**
```bash
docker compose -f docker-compose.fullstack.yml up --build
# API:  http://localhost:8000/api/health
# UI:   http://localhost:5173
```

**With real MQTT broker:**
```bash
docker compose -f docker-compose.mqtt.yml up -d        # start Mosquitto
INGESTION_MODE=mqtt uvicorn app.main:app --reload      # backend reads from broker
python sim/simulator.py                                 # publish telemetry
```

---

## ML models

Three trained models live in `models/`:

| Model | File | Predicts |
|---|---|---|
| Remaining Useful Life | `rul_best.pkl` | Laps until component failure |
| Battery SOC drop | `soc_drop_v1.pkl` | % SOC lost per lap |
| Brake wear | `brake_wear_v1.pkl` | Pad fraction worn per lap |

To retrain from fresh simulation data:
```bash
python scripts/simulate_laps.py   # generate data/lap_data.csv
python scripts/train_models.py    # train + save to models/
python scripts/predict_latest.py  # verify predictions
```

To export to ONNX:
```bash
python scripts/convert_to_onnx.py
python scripts/test_onnx.py
```

---

## API reference

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/telemetry/latest` | Latest telemetry snapshot |
| GET | `/api/telemetry/history?limit=300` | Rolling history |
| GET | `/api/alerts/latest` | Active alert codes |
| POST | `/api/predictions/run` | Trigger ML inference |
| GET | `/api/predictions/latest` | Last inference result |
| GET | `/api/models/status` | Model registry status |
| WS | `/ws/telemetry` | Live telemetry stream |

WebSocket envelope:
```json
{ "type": "telemetry.update", "payload": { "speed_kph": 142.3, ... } }
{ "type": "snapshot",         "data":    { "speed_kph": 142.3, ... } }
```

---

## AWS deployment

### Prerequisites
- AWS CLI configured
- Terraform >= 1.6
- Docker

### Deploy infrastructure
```bash
cd deploy/terraform
terraform init
terraform apply -var="aws_region=eu-west-1"
```

### Fill secrets (GitHub → Settings → Secrets)
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPO
ECS_CLUSTER
ECS_SERVICE
S3_BUCKET
CF_DIST_ID
```

Push to `main` — GitHub Actions builds the Docker image, pushes to ECR, updates ECS, builds the React app, syncs to S3, and invalidates CloudFront.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `INGESTION_MODE` | `sim` | `sim` / `mqtt` / `csv` |
| `MQTT_HOST` | `localhost` | MQTT broker host |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `uvtwin/telemetry/#` | Subscribe topic |
| `LATEST_CSV` | `data/latest.csv` | CSV fallback path |
| `MODELS_DIR` | auto-detected | Path to `*.pkl` model files |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |

---

## Testing

```bash
# Backend
cd backend && pytest --tb=short -q

# Frontend
cd frontend && npm run test

# Full verification
pwsh scripts/verify_fullstack.ps1

# E2E (requires running stack)
npx playwright test
```

---

## Project structure

```
.
├── backend/                  FastAPI service
│   ├── app/
│   │   ├── api/              REST + WebSocket endpoints
│   │   ├── ingestion/        MQTT consumer + CSV fallback
│   │   ├── ml/               Model registry + predict service
│   │   ├── realtime/         EventBus + WebSocket manager
│   │   ├── repositories/     In-memory telemetry buffer
│   │   ├── schemas/          Pydantic models
│   │   ├── services/         Alert rules + normalizer
│   │   └── simulation/       Built-in physics simulator
│   └── tests/
├── frontend/                 React + Vite SPA
│   └── src/
│       ├── components/       KPI grid, charts, alerts, 3D twin
│       ├── context/          WebSocket realtime context
│       ├── pages/            Overview, Telemetry, Predictions, 3D, Diagnostics
│       └── state/            Realtime state reducer
├── deploy/
│   ├── aws/                  ECS task def + runbook notes
│   └── terraform/            Full IaC (ECR, ECS, ALB, S3, CloudFront)
├── .github/workflows/        CI (test+build) + CD (ECR→ECS, S3→CF)
├── models/                   Trained joblib + ONNX artifacts
├── sim/                      Physics-based vehicle simulator
├── scripts/                  Train, predict, explain, MQTT tools
└── docker-compose.fullstack.yml
```
