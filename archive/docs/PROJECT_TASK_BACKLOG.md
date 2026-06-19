# Project Task Backlog

## Priority 1: Sprint Zero Execution (Environment & Baseline Validation)
- [ ] **Task 1.1**: Verify current branch status and document starting point.
- [ ] **Task 1.2**: Verify dependency versions and environment setup for both ARGUS and NEXUS.
- [ ] **Task 1.3**: Execute baseline NEXUS and verify behavior/capture baseline metrics.
- [ ] **Task 1.4**: Execute baseline ARGUS and verify behavior/capture baseline metrics.
- [ ] **Task 1.5**: Generate Sprint Zero Completion Report (including issues found, commands executed, files created, and Go/No-Go for Sprint One).

## Priority 2: Vision Bridge & Microservice Infrastructure (Sprint One)
- [ ] **Task 2.1**: Implement `inference_server.py` as a microservice wrapper for ARGUS.
- [ ] **Task 2.2**: Implement `vision_bridge.py` using ZeroMQ/gRPC to handle communication between the ARGUS Service and NEXUS Service.

## Priority 3: MULDE Retraining for Traffic
- [ ] **Task 3.1**: Implement `train_mulde_traffic.py` targeting AI City Challenge and UA-DETRAC normals.
- [ ] **Task 3.2**: Execute MULDE retraining and calibrate anomaly threshold.
- [ ] **Task 3.3**: Validate retrained MULDE on AI City anomalies and DoTA CCTV subset.

## Priority 4: D3QN & Environment Modification (NEXUS Redesign)
- [ ] **Task 4.1**: Implement `env_anomaly.py` to support synthetic anomaly surrogates during SUMO RL training.
- [ ] **Task 4.2**: Modify `d3qn_multimodal.py` state space representation to fuse traffic variables and anomaly signals.
- [ ] **Task 4.3**: Modify the reward function to account for incident-responsive behavior.
- [ ] **Task 4.4**: Update the replay buffer structure to support multimodal inputs.

## Priority 5: Core Online RL Training
- [ ] **Task 5.1**: Integrate D3QN multimodal agent with `env_anomaly.py`.
- [ ] **Task 5.2**: Execute online RL training in SUMO.
- [ ] **Task 5.3**: Checkpoint models and monitor convergence.

## Priority 6: Evaluation & Stress Testing
- [ ] **Task 6.1**: Run experiments comparing Fixed-Time, NEXUS, and NEXUS+ARGUS.
- [ ] **Task 6.2**: (Optional) Use CARLA solely for stress testing and visual realism experiments.
- [ ] **Task 6.3**: Aggregate experimental data, generate ablations, run statistical tests, and compile evaluation metrics.

## Priority 7: Paper Writing & Code Release Readiness
- [ ] **Task 7.1**: Translate experimental results into the Paper Structure (Introduction, Proposed Framework, Methodology, Experiments).
- [ ] **Task 7.2**: Prepare repository for potential open-source release (clean up branches, resolve technical debt).
- [ ] **Task 7.3**: Conduct Reviewer Attack Simulation and document failure recovery strategies.
