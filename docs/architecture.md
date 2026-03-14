# NEXUS-ATMS — System Architecture

## Overview

The **NEXUS Adaptive Traffic Management System** is a modular, AI-driven platform
for urban traffic optimisation.  It provides two execution modes:

1. **SUMO mode** — uses the SUMO microsimulator (TraCI) with a 4×4 grid (16 junctions)
2. **Standalone mode** — first-principles physics engine (`control/traffic_env.py`)
   with Poisson arrivals, queuing-theory departures, and NEMA-style signal phasing

Both modes feed the same RL agents, dashboards, and analytics pipeline.

---

## High-Level Module Diagram

```
┌──────────────────────── LAYER 1 · DATA INGESTION ──────────────────────┐
│  📷 Traffic Cameras      🚗 SUMO / TraCI      📡 IoT MQTT Sensors    │
│     (YOLOv8 + OpenCV)      (4×4 grid)           (loop/radar/env)      │
│  🌦 Weather API          🚇 Transit GTFS                              │
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
┌──────────────────────── LAYER 2 · AI / ML ENGINE ──────────────────────┐
│  🧠 DQN / PPO / A2C       📈 LSTM Predictor     🔍 Anomaly Detector  │
│     (Stable-Baselines3)      (Seq2Seq Enc-Dec)     (IForest+AE+Zscore)│
│  🎯 YOLOv8 Vision         🔮 Explainable AI (XAI)                    │
│     (detect/track/speed)     (SHAP, Grad-Saliency, Permutation)       │
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
┌─────────────────── LAYER 3 · SPECIALTY MODULES ────────────────────────┐
│  🚑 Emergency Engine       🌱 Carbon Engine     💬 NL Command Parser  │
│     (A* path · cascade)      (ISO 14064 · ESG)    (spaCy + regex)     │
│  🔐 Cybersecurity          🔧 Road Maintenance  🚶 Pedestrian Safety  │
│     (HMAC · rate limit)      (hard-brake AI)       (mediapipe/synth)   │
│  🔁 Counterfactual         🔊 Voice Broadcast                         │
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
┌───────────────────────── LAYER 4 · BACKEND ────────────────────────────┐
│  ⚡ FastAPI + WebSockets   🗄 JSON/NPZ Logs      📋 Audit Trail       │
│     30+ REST endpoints       Model checkpoints      Decision logging   │
│     1 Hz live stream (WS)    TensorBoard             Reversible ops    │
└───────────────────────────────┬────────────────────────────────────────┘
                                ▼
┌───────────────────── LAYER 5 · PRESENTATION ───────────────────────────┐
│  🖥 Authority Dashboard    👥 Citizen Portal     📊 AI Analytics       │
│     Signal grid · Twin       Route predictor       LSTM/Anomaly/XAI    │
│     Alerts · Carbon · NL     Forecast bars         Comparison table    │
│     Emergency · Security     Impact stats          Loss chart          │
│  📐 System Architecture                                                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Environment Layer

### SUMO Mode (`src/envs/sumo_env.py`, `src/envs/multi_agent_env.py`)

| Property | Single Intersection | Multi-Agent (4×4) |
|----------|--------------------|--------------------|
| State dim | 13 | 21 per junction (336 total) |
| Action | Discrete(2) {keep, switch} | MultiDiscrete(16×n_phases) |
| Reward | −α·wait − β·queue + γ·throughput | cooperative + local |
| Dependency | SUMO TraCI | SUMO TraCI |

### Standalone Mode (`control/traffic_env.py`)

| Property | Value |
|----------|-------|
| State dim | 26 |
| Action | Discrete(n_phases) — NEMA 4-phase |
| Arrival model | Poisson (time-of-day demand curve) |
| Departure | Queueing-theory (saturation flow × green fraction) |
| Reward | multi-objective (throughput, delay, stops, emissions, safety) |
| External dep | None |

---

## Reward Function Design

The **combined** reward balances multiple objectives:

```
R(t) = −0.5 · W(t)/200  −  0.3 · Q(t)/50  +  0.2 · T(t)/10
```

| Symbol | Meaning | Normalisation |
|--------|---------|---------------|
| W(t) | Total waiting time across approaches | ÷200 seconds |
| Q(t) | Total halting vehicles | ÷50 vehicles |
| T(t) | Vehicles that exited the network | ÷10 vehicles |

The standalone environment adds emissions and stop-penalty terms.

---

## RL Agents

| Agent | Algorithm | Library | Fallback |
|-------|-----------|---------|----------|
| DQNAgent | Deep Q-Network | SB3 | CPU if no CUDA |
| PPOAgent | Proximal Policy Optimization | SB3 | CPU if no CUDA |
| RLController | PPO (configurable) | SB3 | Webster heuristic |

Network architecture: `Input(state_dim) → FC(256) → FC(256) → Output(n_actions)`

---

## AI / ML Pipeline

| Component | File | Model | Output |
|-----------|------|-------|--------|
| LSTM Predictor | `prediction/lstm_predictor.py` | Seq2Seq bidir LSTM (Enc-Dec) | 30-min traffic forecast |
| Anomaly Detector | `prediction/anomaly_detector.py` | Z-score + IQR + rate-of-change | Real-time anomaly alerts |
| ML Anomaly | `prediction/ml_anomaly_detector.py` | IsolationForest + Autoencoder | F1-scored anomaly detection |
| Explainable AI | `src/explainability/explainer.py` | SHAP, Gradient Saliency, Permutation | Feature importance, decision explanations |

---

## Vision Pipeline

```
Camera Frame → YOLOv8 Detector → IOU Tracker → Speed Estimator
                                       ↓
                              Zone Counter → Queue Vector
                                       ↓
                              Incident Detector → Alerts
```

Fallback chain: YOLOv8 → OpenCV DNN MobileNet-SSD → Synthetic

---

## IoT & Sensor Fusion

```
Loop Detectors ─┐
Radar/LiDAR ────┼──→  SensorFusion  ──→  IntersectionSnapshot
Vision counts ──┘       (EMA + weights)     ├── to_feature_vector() → RL obs
                                             └── snapshot() → Dashboard
```

Published via in-process bus or MQTT (`iot/mqtt_client.py`).

---

## Dashboard & API

**Backend:** FastAPI (`dashboard/backend/main.py`)

| Category | Endpoints |
|----------|-----------|
| Core | `/api/status`, `/api/snapshot`, `/api/intersections` |
| Control | `/api/signal/override`, `/api/emergency/*` |
| Carbon | `/api/carbon/today`, `/api/carbon/certificate` |
| Security | `/api/security/events` |
| Maintenance | `/api/maintenance/orders` |
| NLP | `/api/nl/command` |
| AI | `/api/ai/status`, `/api/ai/lstm/results`, `/api/ai/anomaly/results` |
| XAI | `/api/ai/xai/importance`, `/api/ai/explain` |
| Comparison | `/api/ai/comparison`, `/api/ai/training-history` |
| WebSocket | `/ws/live` (1 Hz junction + traffic + carbon + emergency) |

**Frontend:** Single-page HTML served from `dashboard/frontend/index.html`
- 4 tabs: Authority Dashboard, Citizen Portal, AI Analytics, System Architecture
- Canvas-based Digital Twin (4×4 animated city at 60 fps)
- Interactive 16-junction Signal Grid
- Chart.js for LSTM loss curves
- WebSocket for real-time KPI updates

---

## Training Pipeline

```
1. Load config (configs/default.yaml)
2. Choose environment:
   └── SUMO available? → SumoEnvironment / MultiAgentSumoEnv
   └── No SUMO?        → TrafficEnvironment (standalone)
3. Create agent (DQN, PPO, or A2C)
4. Train with SB3 callbacks:
   ├── EvalCallback (periodic evaluation)
   ├── CheckpointCallback (save checkpoints)
   └── Progress bar
5. Save best + final model
6. Generate TensorBoard logs
7. Run LSTM training pipeline (scripts/train_lstm.py)
8. Run ML anomaly training (prediction/ml_anomaly_detector.py --generate)
9. Generate XAI report (src/explainability/explainer.py)
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / AMD Ryzen 5 | Ryzen 7+ |
| GPU | Not required (CPU fallback) | CUDA GPU (RTX 2050+) |
| RAM | 8 GB | 16 GB |
| VRAM | — | 4 GB+ |
| Python | 3.10+ | 3.13 |

Actual project hardware: ASUS VivoBook 15 Pro, AMD Ryzen 5 5600H,
NVIDIA RTX 2050 (4.3 GB VRAM), CUDA 12.4, PyTorch 2.6.0+cu124.

---

## Algorithm Comparison

| Feature | DQN | PPO |
|---------|-----|-----|
| Policy type | Value-based | Policy gradient |
| Sample efficiency | Lower | Higher |
| Stability | Moderate | High |
| Replay buffer | Yes (100K) | No (on-policy) |
| Recommended for | Simple experiments | Production use |
