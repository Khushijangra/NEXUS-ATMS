# Sprint One Implementation Plan

## 1. Objective
Design and implement the integration bridge between the isolated ARGUS perception subsystem and the NEXUS-ATMS reinforcement learning control subsystem. The bridge must allow the RL agent to receive real-time anomaly scores without breaking environment isolation or blocking simulation execution.

## 2. Architecture Overview
The integration introduces a loosely coupled IPC (Inter-Process Communication) architecture using ZeroMQ. 

1. **ARGUS Subsystem (Server)**: A new `inference_server.py` script runs within the ARGUS environment context. It keeps the VideoMAE backbone and MULDE scorer in memory, listening on a ZeroMQ `REP` socket.
2. **NEXUS Subsystem (Client)**: A new `vision_bridge.py` acts as a ZMQ `REQ` client. It polls the inference server during the RL step.
3. **Environment**: `env_anomaly.py` forks the existing SUMO environment to inject synthetic traffic incidents that correspond to the visual anomalies.
4. **Agent**: `d3qn_multimodal.py` wraps the existing D3QN logic to accept a dynamically expanded state vector containing both standard traffic metrics and the new anomaly data.

## 3. Workflow
1. **Reset Phase**: `env_anomaly.py` resets SUMO and clears the `vision_bridge` buffers.
2. **Step Phase**: 
   - `env_anomaly.py` advances the SUMO simulation.
   - It queries `vision_bridge.py` for the current visual anomaly context.
   - `vision_bridge.py` queries `inference_server.py` via ZeroMQ (with timeout protection).
   - The combined `traffic_features` + `anomaly_score` + `anomaly_flag` state is constructed.
   - `d3qn_multimodal.py` computes the next action based on the expanded state.
