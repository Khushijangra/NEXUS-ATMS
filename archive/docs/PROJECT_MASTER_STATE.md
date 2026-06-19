# Project Master State

## Project Objective
Develop a Vision-Aware Incident-Responsive Adaptive Traffic Signal Control System by integrating **NEXUS-ATMS** (Urban Congestion Prediction + RL Traffic Signal Control) and **ARGUS Stream** (VideoMAE + MULDE Anomaly Detection).

## Final Dataset Strategy
* **ARGUS Normal Traffic Representation Learning**: UA-DETRAC + AI City Challenge Track 4 (normals)
* **ARGUS Anomaly Validation & External Testing**: AI City Challenge Track 4 (anomalies) + DoTA CCTV subset
* **SUMO Vehicle Behavior Calibration**: inD dataset
* **Road Network Generation**: OpenStreetMap
* **Traffic Demand Calibration**: Public traffic counts
* **CARLA**: Optional (reserved only for stress testing and visual realism experiments; strictly NOT part of the core RL training loop).

## Final Architecture
* **Video Backbone**: VideoMAE-v2 (Frozen / Keep unchanged)
* **Anomaly Scorer**: MULDE (Retrained for traffic anomalies)
* **Traffic Forecasting**: LSTM (Keep unchanged)
* **Traffic Controller**: D3QN (Modified to be multimodal)
* **State Space**: Expanded to include Traffic variables + ARGUS anomaly information
* **RL Training Strategy**: Online RL within SUMO environment (Synthetic anomaly surrogate during RL training, real ARGUS inference during validation/stress testing).
* **System Design**: Microservice separation (ARGUS Service + NEXUS Service) connected via Vision Bridge, communicating via ZeroMQ / gRPC.

*Discarded Ideas:* Offline RL, end-to-end VideoMAE inside the RL loop, positioning the paper purely as a video anomaly detection paper.

## Repository Modifications Planned
* `env_anomaly.py` (SUMO environment modified for anomalies)
* `d3qn_multimodal.py` (D3QN modification for expanded state representation)
* `vision_bridge.py` (ZeroMQ/gRPC bridge between services)
* `inference_server.py` (ARGUS microservice wrapper)
* `train_mulde_traffic.py` (MULDE retraining script)
* *General Modifications*: D3QN state representation, reward function, replay buffer structure.

## Paper Structure
* **Introduction**
* **Proposed Framework**: Layered Architecture with complete pipeline, Module Dependency Graph
* **Methodology**: Vision Layer, Prediction Layer, Fusion Layer, RL Control Layer
* **Experiments**: Fixed-Time, NEXUS, NEXUS+ARGUS

## Current Development Phase
**SYSTEM SPECIFICATION FREEZE / EXECUTION MODE**
We have completed Phase 2 (Architecture Freeze) and are currently entering Sprint Zero Execution. The planning loop is concluded; focus is strictly on engineering execution.

## Completed Milestones
* ✓ Problem formulation
* ✓ Dataset strategy selection
* ✓ Workflow strategy
* ✓ Publication positioning
* ✓ High-level architecture / Architecture Freeze
* ✓ Feasibility analysis
* ✓ Implementation Blueprint
* ✓ Repository Gap Analysis

## Pending Milestones
* Sprint Zero: Baseline validation and setup verification
* Sprint One: Module modifications and Vision Bridge implementation
* Sprint Two: Core training execution (MULDE & D3QN)
* Sprint Three: Experiment Execution & Ablations
* Sprint Four: Paper Writing

## Sprint Zero Status
**Status: Pending / Executing**
Focus is on execution, validation, and repository readiness.
Objectives:
1. Verify current branch status.
2. Verify dependency versions.
3. Verify environment setup.
4. Verify baseline NEXUS execution.
5. Verify baseline ARGUS execution.
6. Generate baseline metrics.
7. Generate Sprint Zero completion report (commands executed, files created, issues found, Go/No-Go for Sprint One).

## Sprint One Objectives
1. Implement `vision_bridge.py` and `inference_server.py`.
2. Modify SUMO environment (`env_anomaly.py`).
3. Implement `d3qn_multimodal.py` with expanded state representation and modified replay buffer.
4. Prepare `train_mulde_traffic.py` pipeline.

## Known Risks
* Changing datasets, reward functions, or workflows after this point is extremely expensive and prohibited.
* Integration complexity between ARGUS and NEXUS via ZeroMQ/gRPC (Vision Bridge).

## Open Action Items
* Execute Sprint Zero objectives.
* Run NEXUS baseline and capture metrics.
* Run ARGUS baseline and capture metrics.
* Draft Sprint Zero completion report.
