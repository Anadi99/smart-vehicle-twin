# Smart Vehicle Twin

[![CI](https://github.com/Anadi99/smart-vehicle-twin/actions/workflows/ci.yml/badge.svg)](https://github.com/Anadi99/smart-vehicle-twin/actions/workflows/ci.yml)
[![Deploy](https://github.com/Anadi99/smart-vehicle-twin/actions/workflows/deploy.yml/badge.svg)](https://github.com/Anadi99/smart-vehicle-twin/actions/workflows/deploy.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-6.0-3178C6?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![AWS](https://img.shields.io/badge/AWS-ECS%20%7C%20S3%20%7C%20CloudFront-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Terraform](https://img.shields.io/badge/Terraform-1.6+-7B42BC?logo=terraform&logoColor=white)](https://terraform.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A production-grade, cloud-native **AI-powered Digital Twin** for a BMW i4 race car.
> Real-time telemetry, ML predictions, anomaly detection, SHAP explainability, and a live 3D visualization — all running on AWS and deployable in one command.

---

## What is this project?

Modern vehicles generate thousands of sensor readings per second. A **Digital Twin** is a real-time virtual replica of a physical asset — in this case, a BMW i4 on a race track. Instead of waiting for a component to fail, the twin continuously monitors health signals, predicts failures before they happen, and recommends pit-stop actions to the team.

This project demonstrates a complete, end-to-end Cloud AI system:

- A **physics-based vehicle simulator** generates realistic telemetry (speed, brake temperature, battery SOC, tire wear, GPS position) at 1 Hz
- A **FastAPI backend** ingests that data via MQTT or CSV, runs it through an ML pipeline, evaluates alert rules, and broadcasts updates to all connected clients over WebSocket
- Three **trained ML models** (Random Forest / XGBoost) predict Remaining Useful Life in laps, battery SOC drop per lap, and brake wear per lap — with SHAP explainability
- An **Isolation Forest** model detects anomalous sensor patterns in real time
- A **React dashboard** renders live KPI cards, multi-series charts, a 3D animated vehicle model, per-lap analytics, and ML prediction panels — all updating without page refresh
- The entire stack is **containerised with Docker**, deployed to **AWS ECS Fargate** behind an ALB, with the frontend on **S3 + CloudFront**, provisioned by **Terraform**, and automated by **GitHub Actions CI/CD**

---

## The Problem It Solves

In motorsport and fleet management, unplanned component failures are expensive and dangerous. Teams currently rely on:

- Manual inspection between sessions
- Reactive maintenance after failure
- Gut-feel decisions about pit timing

This system replaces that with:

- **Predictive maintenance** — know 5 laps before the brake pad hits the wear limit
- **Real-time anomaly detection** — catch sensor spikes and unusual patterns instantly
- **Data-driven pit strategy** — ML models recommend the optimal pit window based on current wear rates
- **Explainable AI** — SHAP values show exactly which sensor reading drove the prediction, building trust with engineers

The same architecture applies directly to EV fleet management, industrial IoT, and autonomous vehicle monitoring.

---

## Live Demo

```
Dashboard:  http://localhost:5173   (after docker compose up)
API docs:   http://localhost:8000/docs
Metrics:    http://localhost:8000/metrics
Grafana:    http://localhost:3000   (admin / svtwin2026)
```

Default login: `admin` / `svtwin2026`

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                                       │
│                                                                              │
│   ┌─────────────────┐  HTTPS   ┌──────────────────────────────────────────┐ │
│   │   CloudFront    │◄────────►│        S3 Bucket                         │ │
│   │   (CDN + WAF)   │          │   React SPA (static assets)              │ │
│   └────────┬────────┘          └──────────────────────────────────────────┘ │
│            │  /api/*  /ws/*                                                  │
│   ┌────────▼────────┐          ┌──────────────────────────────────────────┐ │
│   │  Application    │◄────────►│         ECS Fargate Task                 │ │
│   │  Load Balancer  │          │  ┌──────────────┐  ┌──────────────────┐  │ │
│   └─────────────────┘          │  │  FastAPI     │  │   ML Pipeline    │  │ │
│                                │  │  + Uvicorn   │  │  rul_best.pkl    │  │ │
│                                │  │  WebSocket   │  │  soc_drop_v1.pkl │  │ │
│                                │  │  EventBus    │  │  brake_wear.pkl  │  │ │
│                                │  │  Sim Loop    │  │  IsolationForest │  │ │
│                                │  └──────────────┘  └──────────────────┘  │ │
│                                └──────────────────────────────────────────┘ │
│                                              ▲                               │
│                                   ┌──────────┴──────────┐                   │
│                                   │   ECR Repository    │                   │
│                                   │   (Docker images)   │                   │
│                                   └─────────────────────┘                   │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  Observability: Prometheus /metrics  →  Grafana dashboards           │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘

IoT Edge / Local
┌──────────────────────────────────────────────────────────────────┐
│  sim/simulator.py  ──MQTT──►  Mosquitto Broker  ──►  Backend     │
│  (BMW i4 physics sim)         (docker-compose.mqtt.yml)          │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Vehicle Sensor / Simulator (1 Hz tick)
  │
  ├─► MQTT publish  ──────────────────────────────────────────────┐
  └─► CSV write (fallback)                                        │
                                                                  ▼
                                                    FastAPI Ingestion Layer
                                                          │
                                              ┌───────────┴────────────┐
                                              ▼                        ▼
                                    TelemetryRepository          Alert Engine
                                    (rolling 500 buffer)         (7 rule types)
                                              │                        │
                                              ▼                        ▼
                                        EventBus.publish()      Anomaly Detector
                                              │                 (Isolation Forest)
                                    ┌─────────┴──────────┐           │
                                    ▼                    ▼            ▼
                             WebSocket broadcast    ML Inference   Drift Monitor
                             (all clients, <250ms)  (RUL/SOC/Wear) (PSI per feature)
                                    │
                                    ▼
                             React Dashboard
                             (KPIs, Charts, 3D, Laps, Predictions)
```

---

## Features

### Real-Time Telemetry
- Physics-based BMW i4 simulator generates 10 sensor channels at 1 Hz: speed, brake temperature, battery SOC, brake pad wear, tire wear, acceleration, GPS coordinates, risk score
- WebSocket stream with automatic **exponential-backoff reconnect** (1s → 30s, 10 retries)
- Rolling 500-sample in-memory buffer with full history API
- Three ingestion modes: built-in simulator, MQTT broker, CSV file polling

### Machine Learning Pipeline
- **Remaining Useful Life (RUL)** — predicts laps until brake pad or battery hits the failure threshold
- **Battery SOC drop** — predicts % battery lost per lap at current driving intensity
- **Brake wear** — predicts pad fraction worn per lap
- **ONNX export** — models exported to `rul_best.onnx` for edge deployment; ONNX Runtime used when available, joblib as fallback
- **SHAP explainability** — top contributing features visualised as a bar chart in the dashboard
- **Isolation Forest anomaly detection** — online warm-up on first 200 samples, then scores every reading
- **Data drift monitoring** — PSI (Population Stability Index) per feature vs training baseline

### Alert Engine
Seven rule-based alert codes with severity levels:

| Code | Severity | Trigger |
|---|---|---|
| `BRAKE_OVERHEAT` | Critical | Brake temp > 95°C |
| `BRAKE_PAD_CRITICAL` | Critical | Pad fraction < 0.20 |
| `LOW_SOC` | Critical | Battery SOC < 15% |
| `RISK_HIGH` | Critical | Risk score ≥ 0.70 |
| `OVERSPEED` | Critical | Speed > 250 km/h |
| `BRAKE_TEMP_HIGH` | Warning | Brake temp > 85°C |
| `BRAKE_PAD_LOW` | Warning | Pad fraction < 0.30 |
| `SOC_WARNING` | Warning | Battery SOC < 20% |
| `RISK_MEDIUM` | Warning | Risk score ≥ 0.40 |

### Dashboard Pages
| Page | Description |
|---|---|
| **Overview** | Live KPI cards, active alerts with pulse animation, telemetry trend charts |
| **Telemetry** | Full 8-metric KPI grid, 3-panel chart layout (speed/temp, SOC/pad, risk/wear) |
| **Predictions** | ML inference results, SHAP bar chart, anomaly status, data drift PSI |
| **3D Twin** | Animated Three.js vehicle orbiting a track oval at real speed, health-colored body |
| **Laps** | Per-lap aggregated analytics table — speed, brake temp, SOC drop, risk |
| **Diagnostics** | System health, ingestion mode, model registry, live WS snapshot |

### Security
- JWT Bearer token authentication (OAuth2 password flow)
- Protected endpoints: `POST /api/predictions/run`, `GET /api/predictions/explain`
- Configurable secret key, expiry, and credentials via environment variables

### Observability
- Prometheus metrics at `/metrics` via `prometheus-fastapi-instrumentator`
- Pre-built Grafana dashboard (request rate, P95 latency, error rate, WS connections)
- CloudWatch Logs integration in ECS task definition

### Infrastructure as Code
- Full Terraform module: ECR, ECS Fargate, ALB, S3, CloudFront, IAM roles, CloudWatch log groups
- One command deploy: `terraform apply`
- GitHub Actions CI/CD: test → build → push to ECR → update ECS → sync S3 → invalidate CloudFront

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Backend language** | Python | 3.11 |
| **API framework** | FastAPI + Uvicorn | 0.11x |
| **Data validation** | Pydantic | v2 |
| **ML models** | scikit-learn (RandomForest, IsolationForest) | latest |
| **ML boost** | XGBoost | optional |
| **Model serialisation** | joblib + ONNX Runtime | latest |
| **Explainability** | SHAP | optional |
| **Auth** | python-jose (JWT) + passlib (bcrypt) | latest |
| **Metrics** | prometheus-fastapi-instrumentator | latest |
| **MQTT** | paho-mqtt | latest |
| **Frontend language** | TypeScript | 6.0 |
| **UI framework** | React | 19 |
| **Build tool** | Vite | 8 |
| **Styling** | Tailwind CSS | v4 |
| **Charts** | Recharts | 3.x |
| **3D rendering** | Three.js + React Three Fiber | 0.183 |
| **Data fetching** | TanStack Query | v5 |
| **Routing** | React Router | v7 |
| **Frontend tests** | Vitest + Testing Library | latest |
| **E2E tests** | Playwright | latest |
| **Load tests** | Locust | latest |
| **Containers** | Docker + docker-compose | latest |
| **Cloud** | AWS ECS Fargate, ALB, ECR, S3, CloudFront | — |
| **IaC** | Terraform | >= 1.6 |
| **CI/CD** | GitHub Actions | — |
| **Code quality** | ruff, mypy, eslint, pre-commit | latest |

---

## Project Structure

```
smart-vehicle-twin/
│
├── backend/                          FastAPI backend service
│   ├── app/
│   │   ├── api/                      REST + WebSocket route handlers
│   │   │   ├── auth.py               POST /api/auth/token (JWT login)
│   │   │   ├── health.py             GET  /api/health
│   │   │   ├── telemetry.py          GET  /api/telemetry/latest|history
│   │   │   ├── alerts.py             GET  /api/alerts/latest
│   │   │   ├── predictions.py        POST /api/predictions/run + explain + anomaly + drift
│   │   │   ├── laps.py               GET  /api/laps
│   │   │   ├── status.py             GET  /api/status
│   │   │   └── ws.py                 WS   /ws/telemetry
│   │   ├── auth/
│   │   │   └── jwt.py                JWT token creation + verification
│   │   ├── ingestion/
│   │   │   ├── mqtt_consumer.py      paho-mqtt subscriber
│   │   │   ├── csv_fallback.py       Async CSV file poller
│   │   │   └── ingestion_runner.py   MQTT → CSV fallback orchestrator
│   │   ├── ml/
│   │   │   ├── predict_service.py    RUL / SOC / wear inference (ONNX + joblib)
│   │   │   ├── model_registry.py     Artifact discovery
│   │   │   ├── anomaly.py            Isolation Forest (online)
│   │   │   ├── drift.py              PSI data drift monitor
│   │   │   └── explainability.py     SHAP / feature importance
│   │   ├── realtime/
│   │   │   ├── event_bus.py          asyncio pub/sub fan-out
│   │   │   └── ws_manager.py         WebSocket connection handler
│   │   ├── repositories/
│   │   │   └── telemetry_repo.py     Rolling 500-sample deque
│   │   ├── schemas/
│   │   │   └── telemetry.py          Pydantic Telemetry model (10 fields)
│   │   ├── services/
│   │   │   ├── alerts.py             Rule engine (9 alert codes)
│   │   │   ├── normalizer.py         Legacy CSV key → canonical field mapping
│   │   │   └── lap_aggregator.py     Per-lap stats aggregation
│   │   ├── simulation/
│   │   │   └── telemetry_sim.py      Built-in BMW i4 physics simulator
│   │   ├── config.py                 Settings (env-var driven)
│   │   ├── dependencies.py           FastAPI DI singletons
│   │   └── main.py                   App factory + lifespan + CORS + Prometheus
│   └── tests/                        33 pytest tests
│
├── frontend/                         React 19 + Vite SPA
│   └── src/
│       ├── components/
│       │   ├── alerts/AlertsRail.tsx  Alert list with icons + severity
│       │   ├── charts/TelemetryCharts.tsx  3-panel Recharts layout
│       │   ├── kpi/KpiGrid.tsx        8 animated KPI cards
│       │   ├── layout/AppShell.tsx    Sidebar + top bar
│       │   ├── status/ConnectionBadge.tsx  WS status indicator
│       │   └── twin/                  Three.js 3D scene + health overlay
│       ├── context/RealtimeContext.tsx  WebSocket provider
│       ├── hooks/useRealtime.ts        Realtime state hook
│       ├── lib/
│       │   ├── api.ts                 fetch wrappers
│       │   ├── auth.ts                JWT token storage + authFetch
│       │   └── wsClient.ts            WS client with exponential backoff
│       ├── pages/                     6 pages
│       └── state/realtimeStore.ts     State reducer
│
├── deploy/
│   ├── aws/                          ECS task def, ALB notes, IoT Core guide
│   ├── monitoring/                   Prometheus config + Grafana dashboard JSON
│   └── terraform/                    Full IaC (main.tf, variables.tf, outputs.tf)
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    Test + build on every push/PR
│       └── deploy.yml                ECR push + ECS deploy + S3 sync on main
│
├── docs/adr/                         4 Architecture Decision Records
├── models/                           Trained .pkl + .onnx artifacts
├── sim/                              Standalone physics simulators
├── scripts/                          Train, predict, explain, MQTT, ONNX tools
├── tests/
│   ├── e2e/smoke.spec.ts             7 Playwright E2E tests
│   └── load/locustfile.py            Locust load test (all endpoints)
│
├── docker-compose.fullstack.yml      Backend + frontend (production images)
├── docker-compose.dev.yml            Hot-reload dev environment
├── docker-compose.mqtt.yml           With Mosquitto MQTT broker
├── docker-compose.monitoring.yml     Prometheus + Grafana
└── .pre-commit-config.yaml           ruff + mypy + pre-commit hooks
```

---

## Getting Started

### Prerequisites

| Tool | Minimum version |
|---|---|
| Python | 3.10 |
| Node.js | 20 |
| Docker | 24 |
| Git | any |

### Option 1 — Docker (recommended, zero setup)

```bash
git clone https://github.com/Anadi99/smart-vehicle-twin.git
cd smart-vehicle-twin

docker compose -f docker-compose.fullstack.yml up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:5173 |
| API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Prometheus metrics | http://localhost:8000/metrics |

Login with `admin` / `svtwin2026`

---

### Option 2 — Local development (hot reload)

**Terminal 1 — Backend:**
```bash
cd backend
pip install -e ".[test]"
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

---

### Option 3 — With real MQTT broker

```bash
# Start Mosquitto
docker compose -f docker-compose.mqtt.yml up -d

# Start backend in MQTT mode
INGESTION_MODE=mqtt uvicorn app.main:app --reload --port 8000

# Run the simulator (publishes to broker)
python sim/simulator.py
```

---

### Option 4 — With Prometheus + Grafana monitoring

```bash
# Start the full stack + monitoring
docker compose -f docker-compose.fullstack.yml up -d
docker compose -f docker-compose.monitoring.yml up -d

# Grafana: http://localhost:3000  (admin / svtwin2026)
# Prometheus: http://localhost:9090
```

---

## ML Models

### Trained Artifacts

Three models are pre-trained and committed to `models/`:

| Model | File | Algorithm | MAE |
|---|---|---|---|
| Remaining Useful Life | `rul_best.pkl` | XGBoost / RandomForest | ~2.1 laps |
| Battery SOC drop | `soc_drop_v1.pkl` | RandomForest | ~0.005% |
| Brake wear | `brake_wear_v1.pkl` | RandomForest | ~0.045 frac |

An ONNX export of the RUL model is also available at `models/rul_best.onnx` for edge deployment.

### Feature Columns

All three models use the same 5 features derived from per-lap aggregations:

```
speed_mean    — average speed over the lap (km/h)
speed_max     — peak speed over the lap (km/h)
temp_mean     — average brake temperature (°C)
temp_max      — peak brake temperature (°C)
duration_sec  — lap duration (seconds)
```

### Retrain from Scratch

```bash
# 1. Generate simulation data
python scripts/simulate_laps.py        # creates data/lap_data.csv (50 laps)

# 2. Train all three models
python scripts/train_models.py         # saves to models/ + model_meta.json

# 3. Verify predictions
python scripts/predict_latest.py       # prints predictions.json

# 4. Explain the RUL model
python scripts/explain_rul.py          # SHAP or feature importance

# 5. Export to ONNX
python scripts/convert_to_onnx.py      # saves rul_best.onnx
python scripts/test_onnx.py            # validates ONNX vs sklearn output
```

### Anomaly Detection

The `IsolationForest` model runs entirely in-process:
- Collects the first 200 telemetry readings as warm-up data
- Trains automatically when warm-up is complete
- Scores every subsequent reading — `score < threshold` → anomaly
- Status visible at `GET /api/anomaly/latest`

### Data Drift

PSI (Population Stability Index) is computed per feature against a stored training baseline:
- PSI < 0.1 → no drift (green)
- PSI 0.1–0.2 → moderate drift (warning)
- PSI > 0.2 → significant drift (alert)
- Status visible at `GET /api/drift/status`

---

## API Reference

### Authentication

```bash
# Get a token
curl -X POST http://localhost:8000/api/auth/token \
  -d "username=admin&password=svtwin2026"

# Use the token
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/predictions/run -X POST
```

### Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/auth/token` | — | Login, returns JWT |
| `GET` | `/api/health` | — | Health check `{"status":"ok"}` |
| `GET` | `/api/telemetry/latest` | — | Latest telemetry snapshot |
| `GET` | `/api/telemetry/history?limit=300` | — | Rolling history (max 5000) |
| `GET` | `/api/alerts/latest` | — | Active alert code list |
| `POST` | `/api/predictions/run` | JWT | Trigger ML inference |
| `GET` | `/api/predictions/latest` | — | Last inference result |
| `GET` | `/api/predictions/explain` | JWT | SHAP / feature importance |
| `GET` | `/api/anomaly/latest` | — | Isolation Forest score |
| `GET` | `/api/drift/status` | — | PSI drift report |
| `GET` | `/api/laps` | — | All per-lap aggregated stats |
| `GET` | `/api/laps/{n}` | — | Single lap stats |
| `GET` | `/api/models/status` | — | Model registry |
| `GET` | `/api/status` | — | System status (ingestion mode, config) |
| `GET` | `/metrics` | — | Prometheus metrics |
| `WS` | `/ws/telemetry` | — | Live telemetry stream |

### WebSocket Protocol

Connect to `ws://localhost:8000/ws/telemetry`

On connect, you immediately receive a snapshot:
```json
{
  "type": "snapshot",
  "data": {
    "speed_kph": 142.3,
    "brake_temp_c": 67.1,
    "battery_soc": 74.2,
    "brake_pad_frac": 0.241,
    "tire_wear_frac": 0.087,
    "accel_mps2": 1.24,
    "lat": 50.3361,
    "lon": 6.9474,
    "risk": 0.12,
    "reasons": "nominal"
  }
}
```

Then every second:
```json
{
  "type": "telemetry.update",
  "payload": { "speed_kph": 155.7, "brake_temp_c": 71.4, ... }
}
```

---

## Environment Variables

### Backend

| Variable | Default | Description |
|---|---|---|
| `INGESTION_MODE` | `sim` | `sim` built-in simulator / `mqtt` broker / `csv` file |
| `MQTT_HOST` | `localhost` | MQTT broker hostname |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `uvtwin/telemetry/#` | MQTT subscribe topic |
| `LATEST_CSV` | `data/latest.csv` | CSV fallback file path |
| `MODELS_DIR` | auto-detected | Absolute path to `*.pkl` / `*.onnx` files |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `JWT_SECRET` | `change-me-in-production` | JWT signing secret — **change in production** |
| `JWT_EXPIRE_MINUTES` | `60` | Token expiry in minutes |
| `ADMIN_USER` | `admin` | Demo admin username |
| `ADMIN_PASS` | `svtwin2026` | Demo admin password — **change in production** |

### Frontend

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE` | `""` | API base URL (empty = same origin via nginx proxy) |

---

## AWS Deployment

### Infrastructure

The Terraform module in `deploy/terraform/` provisions:

- **ECR** — private Docker registry for the backend image
- **ECS Fargate** — serverless container (256 CPU / 512 MB)
- **ALB** — Application Load Balancer with WebSocket support
- **S3** — static hosting bucket for the React SPA
- **CloudFront** — CDN with OAC, SPA fallback (403/404 → index.html), API/WS forwarding
- **IAM** — ECS task execution role with least-privilege policy
- **CloudWatch** — log group with 7-day retention

### Deploy Steps

```bash
# 1. Provision infrastructure
cd deploy/terraform
terraform init
terraform apply -var="aws_region=eu-west-1"

# Note the outputs:
#   cloudfront_url      = https://xxxx.cloudfront.net
#   ecr_repository_url  = 123456789.dkr.ecr.eu-west-1.amazonaws.com/svt-backend
#   s3_bucket           = svt-frontend-eu-west-1

# 2. Add GitHub Secrets (Settings → Secrets → Actions)
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
#   ECR_REPO, ECS_CLUSTER, ECS_SERVICE, S3_BUCKET, CF_DIST_ID

# 3. Push to main — CI/CD does the rest automatically
git push origin main
```

### CI/CD Pipeline

**On every push / PR:**
```
backend tests (pytest) → frontend lint → frontend tests (vitest) → frontend build
```

**On push to main:**
```
docker build → ECR push → ECS task update → wait for stability
→ npm build → S3 sync → CloudFront invalidation
```

### AWS IoT Core (Production Ingestion)

For production, replace Mosquitto with AWS IoT Core:

```bash
INGESTION_MODE=mqtt
MQTT_HOST=<endpoint>.iot.<region>.amazonaws.com
MQTT_PORT=8883
```

See `deploy/aws/iot-core-notes.md` for full TLS certificate setup.

---

## Testing

### Backend (33 tests)

```bash
cd backend
pytest --tb=short -v
```

| Test file | What it covers |
|---|---|
| `test_health.py` | Health endpoint returns 200 + `{"status":"ok"}` |
| `test_telemetry_api.py` | Latest + history endpoints, limit validation |
| `test_alert_rules.py` | All 9 alert codes trigger at correct thresholds |
| `test_normalizer.py` | Legacy CSV keys + canonical keys + pro_sim keys |
| `test_ingestion_fallback.py` | MQTT → CSV fallback mode switching |
| `test_predictions_api.py` | Auth flow, inference, anomaly, drift, explain |
| `test_laps_api.py` | Laps list + 404 for unknown lap |
| `test_ws_telemetry.py` | WebSocket snapshot on connect |

### Frontend (5 tests)

```bash
cd frontend
npm run test
```

Covers: KPI grid rendering, alerts rail, health overlay logic, realtime store reducer, app shell navigation.

### E2E (Playwright)

```bash
# Requires running stack on port 5173
npx playwright test
```

7 smoke tests: page load, nav links, KPI cards, charts, predictions button, 3D twin, diagnostics.

### Load Test (Locust)

```bash
pip install locust
locust -f tests/load/locustfile.py --host http://localhost:8000
# Open http://localhost:8089
```

Simulates concurrent users hitting all endpoints with realistic weights.

### Full Verification

```bash
pwsh scripts/verify_fullstack.ps1
```

Runs backend tests + frontend lint + frontend tests + frontend build in sequence.

---

## Code Quality

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks: `ruff` (lint + format), `mypy` (type check), trailing whitespace, YAML/JSON validation, private key detection.

---

## Architecture Decisions

Four Architecture Decision Records are in `docs/adr/`:

| ADR | Decision | Rationale |
|---|---|---|
| [001](docs/adr/001-fastapi-over-django.md) | FastAPI over Django/Flask | Native async, Pydantic v2, zero-boilerplate WebSocket |
| [002](docs/adr/002-websocket-eventbus.md) | In-process asyncio EventBus | Zero external deps, fully async, trivially testable |
| [003](docs/adr/003-ecs-fargate-over-ec2.md) | ECS Fargate over EC2 | No instance management, pay-per-task, native ALB WS support |
| [004](docs/adr/004-jwt-auth.md) | Stateless JWT auth | No session store, standard OAuth2, Swagger UI compatible |

---

## Simulator Details

The built-in simulator (`backend/app/simulation/telemetry_sim.py`) models:

- **Speed profile** — sinusoidal with braking zones at 18–24%, 52–58%, 82–86% of lap distance
- **Brake temperature** — rises in braking zones, cools exponentially on straights
- **Battery drain** — quadratic with speed + linear with positive acceleration
- **Brake pad wear** — proportional to braking intensity and speed
- **Tire wear** — proportional to speed and lateral load
- **GPS** — circular track orbit near Nürburgring (50.3357°N, 6.9470°E)
- **Risk score** — composite of brake temp, pad wear, SOC, and speed thresholds
- **Failure injection** — 1-in-300 chance of random spike event per tick

The standalone simulator (`sim/simulator.py`) writes to `data/latest.csv` and `data/history.csv` and optionally publishes to an MQTT broker — compatible with the original Streamlit dashboard.

---

## Roadmap

- [ ] TimescaleDB / InfluxDB persistence (replace in-memory buffer)
- [ ] Multi-vehicle support (vehicle ID in telemetry schema)
- [ ] AWS IoT Core TLS integration in MqttConsumer
- [ ] Model retraining pipeline (triggered by drift alert)
- [ ] Redis pub/sub adapter for multi-replica ECS scaling
- [ ] Mobile-responsive dashboard layout
- [ ] Dark/light theme toggle

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes — ensure `pytest` and `npm run test` pass
4. Open a pull request — the PR template will guide you

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

**Anadi** — Cloud AI Developer

- GitHub: [@Anadi99](https://github.com/Anadi99)
- Project: [smart-vehicle-twin](https://github.com/Anadi99/smart-vehicle-twin)

---

<div align="center">

Built with FastAPI · React · AWS · Terraform · scikit-learn · Three.js

*A complete Cloud AI Developer portfolio project*

</div>
