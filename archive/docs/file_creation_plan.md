# File Creation Plan

## 1. `argus_stream_extracted/argus stream A/scripts/inference_server.py`
- **Responsibility**: Load VideoMAE + MULDE checkpoint once into memory. Bind ZeroMQ `REP` socket. Parse incoming JSON requests, compute or fetch the anomaly score for the requested frame, and reply with JSON.
- **Dependencies**: `zmq`, `json`, `torch`, `VideoMAEFeatureExtractor`, `MULDEScorer`.

## 2. `ai/vision/vision_bridge.py`
- **Responsibility**: Act as a fault-tolerant client to `inference_server.py`. Implement a ZeroMQ `REQ` socket with explicit timeouts. Provide a clean Python interface `get_anomaly_context()` to the RL environment.
- **Dependencies**: `zmq`, `json`.

## 3. `ai/envs/env_anomaly.py`
- **Responsibility**: Fork of `ai/envs/sumo_env.py`. Manage SUMO TraCI connection. In addition to normal traffic routing, periodically inject synthetic incidents (e.g., stopping a vehicle) mapped to the anomaly timeline. Append the bridge's anomaly output to the standard state vector.
- **Dependencies**: `gymnasium`, `traci`, `vision_bridge.py`.

## 4. `ai/rl/d3qn_multimodal.py`
- **Responsibility**: Inherit or wrap the existing `DQNAgent`. Override the initialization to accept the `state_dim + 2` input space. Ensure the ReplayBuffer and target network synchronisation remain intact with the larger dimensional arrays.
- **Dependencies**: `torch`, `numpy`, existing D3QN network definitions.
