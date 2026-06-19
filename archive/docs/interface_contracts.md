# Interface Contracts

## 1. ZeroMQ API Contract (Inference Server <-> Vision Bridge)

**Communication Pattern**: ZeroMQ `REQ` / `REP`
**Port**: `tcp://127.0.0.1:5555`

### Request Payload (Client -> Server)
```json
{
  "action": "get_score",
  "timestamp": 12.5,
  "frame_id": 375,
  "context": "avenue_stream_a"
}
```
*(Note: If operating on pre-extracted features or synthetic surrogates, the `frame_id` or `timestamp` maps to the corresponding sequence index).*

### Response Payload (Server -> Client)
```json
{
  "status": "success",
  "anomaly_score": 0.892,
  "anomaly_flag": 1,
  "incident_type": "stopped_vehicle",
  "processing_time_ms": 14.2
}
```
*Timeout Handling*: If the server fails to respond within 50ms, the `REQ` socket times out, and the bridge returns a safe fallback `(anomaly_score=0.0, anomaly_flag=0)`.

## 2. Multimodal State Contract (Environment <-> RL Agent)

**Existing Traffic State (`S_traffic`)**:
Dimensions: `N` (e.g., 21 per junction in 4x4 grid)
Contains: Queue lengths, waiting times, current phase.

**Anomaly State (`S_anomaly`)**:
Dimensions: `2`
Contains:
1. `anomaly_score` (Continuous float: `[0.0, 1.0]`)
2. `anomaly_flag` (Discrete int: `{0, 1}`)

**Expanded State (`S_multimodal`)**:
Dimensions: `N + 2`
Shape: Concat(`S_traffic`, `S_anomaly`)

The `d3qn_multimodal.py` agent will expect an observation space of `Box(low=0, high=inf, shape=(N+2,), dtype=np.float32)`.
