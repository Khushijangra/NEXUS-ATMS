# Sprint One Risk Update

## Identified Risks & Mitigations

### 1. ZeroMQ Blocking the RL Simulation (High Risk)
- **Risk**: If the `inference_server.py` crashes, hangs, or drops a message, the `zmq.REQ` socket in `vision_bridge.py` will block indefinitely waiting for a response. This will freeze the entire SUMO simulation and RL training loop.
- **Mitigation**: The `vision_bridge.py` must be designed with strict timeout protection using `zmq.Poller`. If a timeout is hit, the bridge must safely return a default `(0.0, 0)` state, discard the socket, and re-initialise the connection so the simulation can continue unimpeded.

### 2. State Dimension Mismatch (Medium Risk)
- **Risk**: Existing D3QN network definitions or standard SUMO replay buffers might hardcode the state dimensions, leading to shape mismatch crashes when the 2 anomaly features are appended.
- **Mitigation**: `d3qn_multimodal.py` must rigorously check the initialised `state_dim` parameter and dynamically construct the first Linear layer size. The `env_anomaly.py` must override the `observation_space` property of the Gym environment accurately.

### 3. GPU Memory Contention (Medium Risk)
- **Risk**: Running SUMO, D3QN, and the VideoMAEv2 backbone simultaneously may exceed available VRAM (4.3 GB on the target RTX 2050 hardware).
- **Mitigation**: Keep the VideoMAE batch size low. Consider running `inference_server.py` in CPU mode if VRAM limits are hit, or use `torch.no_grad()` and FP16 half-precision strictly.

### 4. Timestamp / Frame ID Synchronisation (Low Risk)
- **Risk**: The synthetic incidents injected in SUMO may not align with the visual anomaly sequence fed by the server.
- **Mitigation**: The `env_anomaly.py` step logic must explicitly track the elapsed simulation time and query the `inference_server` using deterministic mapping parameters.
