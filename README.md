# NEXUS-ATMS

AI-Driven Urban Congestion Management and Adaptive Traffic Signal Control.

NEXUS-ATMS is a modular traffic intelligence platform that combines reinforcement learning, computer vision, forecasting, anomaly detection, and operational APIs to reduce congestion and improve city-scale traffic decisions.

## Problem

Urban traffic is dynamic, non-stationary, and multi-objective. Fixed-time signals cannot react to:

- sudden queue spikes
- emergency vehicle priority requirements
- changing demand patterns across junctions
- incident-related instability

## Solution

NEXUS-ATMS provides a unified stack that:

- senses traffic through CV and IoT-like streams
- predicts short-horizon flow with sequence models
- detects anomalies with statistical + ML methods
- optimizes signal behavior with RL agents
- exposes real-time controls and analytics through FastAPI and dashboard UI

## Architecture

### Core Layers

1. Ingestion: camera frames, synthetic/live runtime streams, sensor fusion
2. AI Engine: RL control, prediction, anomaly detection, explainability
3. Control + Safety: emergency corridor, security validation, maintenance logic
4. Service Layer: FastAPI REST + WebSocket endpoints
5. Presentation: frontend dashboard and reports

### Final Module Layout

- backend: API runtime, app entrypoint, backend services
- ai/rl: DQN, PPO, D3QN, graph coordination
- ai/envs: SUMO single-intersection and multi-agent environments
- ai/vision: detector, tracker, counter, renderer, geo-mapper, incidents
- ai/prediction: LSTM forecasting
- ai/anomaly: rule-based and ML anomaly detection
- ai/explainability: XAI analysis pipeline
- ai/utils: metrics, logging, training visualization helpers
- control: traffic control orchestration and signal optimization logic
- iot: simulator, MQTT abstraction, fusion logic
- frontend: dashboard assets
- scripts: benchmarking, validation, report generation, demo tooling
- config, configs: runtime and scenario configuration
- docs: architecture, benchmarks, migration and implementation notes

## Data Flow

Input -> CV/IoT ingest -> Feature state assembly -> AI inference/control -> Backend API/WebSocket -> Dashboard + reports

Detailed path:

1. camera/simulated stream enters ai.vision and iot modules
2. fused state + metrics are built per junction
3. ai.rl policy and ai.prediction forecasts are computed
4. ai.anomaly + safety modules flag incidents and risks
5. backend.main publishes REST responses and live WS updates
6. frontend renders operator view and AI analytics

## Setup

### 1) Create environment

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Optional deployment runtime deps

```bash
pip install -r requirements-deploy.txt
```

### 3) Verify GPU/stack (optional)

```bash
python scripts/check_gpu.py
```

## Run

### Backend (recommended)

```bash
python backend/main.py
```

### Full demo helper

```bash
python run_demo.py
```

### Realtime stack helper

```bash
python scripts/start_realtime_stack.py --backend yolo --device cpu
```

## Training and Evaluation

```bash
python train.py --agent d3qn --timesteps 50000
python evaluate.py --agent d3qn --model models/<run>/best/best_model.pt
```

Benchmark suite:

```bash
python scripts/benchmark_d3qn_suite.py --config configs/default.yaml --timesteps 50000
```

## Dashboard and API

- Dashboard: http://localhost:8000
- API docs: http://localhost:8000/docs
- Live socket: /ws/live

## Key Features

- RL signal optimization (DQN, PPO, custom D3QN)
- graph-based multi-junction coordination support
- CV vehicle detection with robust fallback modes
- LSTM-based traffic forecasting
- statistical + ML anomaly detection
- emergency corridor support for priority vehicles
- explainability endpoint support for policy decisions
- report generation and benchmark gating scripts

## Repository Notes

- Legacy hybrid folders were removed; this repository now uses the final module layout.
- Backend single source of truth is backend/main.py.
- AI single source of truth is under ai/.

## License

This project is intended for academic/research demonstration and engineering showcase use.
