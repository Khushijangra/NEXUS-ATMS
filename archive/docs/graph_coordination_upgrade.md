# Graph Coordination Upgrade Plan

## Architecture

Input pipeline:
1. Sensor fusion and runtime state snapshots
2. Graph state builder
3. Graph coordinator message passing
4. Per-junction D3QN inference
5. Safety shield and hold constraints
6. Control dispatch

Graph placement:
- The graph layer sits between fused state and policy inference.
- Local state remains primary signal.
- Neighbor context is additive through concatenation.

State composition:
- Local state: existing per-junction observation.
- Global context: neighbor mean message.
- Enhanced state: concat(local, neighbor_mean).

## Execution Order

1. Keep graph mode OFF and validate baseline parity.
2. Enable graph mode in multi-agent path only.
3. Run Graph OFF vs Graph ON benchmark.
4. Generate A/B report from benchmark output.
5. Decide promotion based on waiting, queue, spillback, stability.

## One-Day Plan

Hour 1:
- Baseline parity check with graph disabled.

Hour 2:
- Graph module smoke checks and runtime fallback validation.

Hour 3:
- Multi-agent benchmark with graph variant enabled.

Hour 4:
- Generate graph A/B report and inspect deltas.

Hour 5:
- Tune only safe knobs: hold steps, graph debug, context dimensions.

Hour 6:
- Repeat benchmark on stress scenario route set.

Hour 7:
- Collect release artifacts and updated report notes.

Hour 8:
- Freeze config profile for next promotion gate.

## Validation Checklist

Graph OFF:
- Training/inference behavior unchanged.
- No shape mismatch and no runtime warnings.

Graph ON:
- No crashes in coordinator path.
- Fallback to local path works if graph path fails.
- No NaNs in D3QN losses.

A/B Quality:
- Waiting time and queue length are non-regressive.
- Spillback rate improves or remains stable.
- Stability metric is non-regressive.

## Failure Modes and Debugging

Failure: adjacency or feature shape mismatch
- Action: enable coordination debug and inspect shape logs.

Failure: graph inference exception at runtime
- Action: verify fallback logs and continue local policy.

Failure: instability or NaN losses
- Action: keep fail_on_nan true; inspect run metadata and history.

Failure: single-intersection regression
- Action: enforce graph_disabled for single-intersection runs.
