# One-Day SOTA-Leaning Execution Plan (NEXUS-ATMS)

This plan is optimized for one day (10-12 focused hours). It does not claim full SOTA across all modules in 24 hours. It maximizes visible model quality, measurable evidence, and judging impact.

## Mission

Ship one high-quality AI upgrade end-to-end with proof:

1. Better vision pipeline (detector quality + real-time performance evidence)
2. RL baseline-vs-agent benchmark evidence
3. Minimal backend hardening for credibility
4. Honest AI labeling (AI vs heuristic)

## Day Outcome (Must-Have Deliverables)

1. `results/yolo_validation.json` (quality + latency evidence)
2. `results/benchmark_d3qn.json` (RL benchmark evidence)
3. `results/one_day_summary.md` (what improved, metrics, what remains)
4. Working demo launch path using `scripts/start_realtime_stack.py`

---

## Phase 0 (30 min): Environment and Baseline Snapshot

### Commands

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you use a specific Python executable, replace `python` with that path.

### Baseline checks

```powershell
python scripts/check_gpu.py
python run_demo.py --dashboard-only
```

Open dashboard at `http://127.0.0.1:8000`.

### Acceptance criteria

1. Dependencies install without blocking errors.
2. Backend starts and dashboard loads.
3. GPU check output saved in your notes.

---

## Phase 1 (2.5 hours): Vision Upgrade + Evidence

Primary target: improve and quantify vehicle perception quality.

### 1. Run YOLO-only validation (already available)

Use a representative traffic video:

```powershell
python scripts/yolo_only_validation.py --video <path_to_video.mp4> --model yolov8n.pt --frames 240 --stride 2 --conf 0.35 > results/yolo_validation.json
```

### 2. Start integrated realtime stack and verify live path

```powershell
python scripts/start_realtime_stack.py --source 0 --backend yolo --device cuda
```

If CUDA is unavailable:

```powershell
python scripts/start_realtime_stack.py --source 0 --backend yolo --device cpu
```

### 3. Capture proof artifacts

1. Screenshot dashboard with live feed and metrics.
2. Keep `results/yolo_validation.json` as benchmark proof.

### Acceptance criteria

1. Validation JSON exists with latency and detection statistics.
2. Live stack runs and displays detections.
3. p95 latency and sampled frames are recorded.

---

## Phase 2 (3 hours): RL Benchmark Evidence

Primary target: show policy value vs baseline with reproducible output.

### Run benchmark suite

```powershell
python scripts/benchmark_d3qn_suite.py --config configs/default.yaml --timesteps 30000
```

Optional graph variant (if stable on your machine):

```powershell
python scripts/benchmark_d3qn_suite.py --config configs/default.yaml --timesteps 30000 --include-graph-variant
```

### Optional focused evaluate run

```powershell
python evaluate.py --model models/<run_name>/best/best_model.zip --agent dqn --n-episodes 10 --report
```

### Acceptance criteria

1. `results/benchmark_d3qn.json` exists.
2. Summary contains waiting time, queue length, and throughput.
3. You can state baseline vs RL deltas with numbers.

---

## Phase 3 (2 hours): Backend Credibility Hardening (Minimum)

Primary target: remove obvious audit red flags.

### Must-do edits

1. Add a simple API key guard for control endpoints in `backend/main.py`.
2. Restrict CORS origin list (remove `*` for non-demo mode).
3. Move hardcoded model paths to environment variables/config lookups.
4. Ensure any "fallback" mode is surfaced in response payload/status.

### Quick verification

```powershell
python -m compileall -q dashboard backend src control iot prediction vision modules scripts
python run_demo.py --dashboard-only
```

### Acceptance criteria

1. Control endpoints require key/token in hardened mode.
2. CORS is no longer wildcard in hardened mode.
3. Backend still starts and serves dashboard.

---

## Phase 4 (1.5 hours): Honest AI Labeling + Final Report

Primary target: align claims with what is truly AI.

### Create one-day report

Create `results/one_day_summary.md` with:

1. What changed today
2. Metrics before vs after
3. Real AI modules vs heuristic modules
4. Known limitations not solved in one day
5. Next 7-day plan

### Suggested structure

```text
# One-Day Upgrade Summary
## 1) Vision Results
## 2) RL Benchmark Results
## 3) Security/Backend Hardening Done
## 4) AI vs Heuristic Truth Table
## 5) Remaining Gaps
## 6) Next 7 Days
```

### Acceptance criteria

1. Report exists and links to `results/yolo_validation.json` and `results/benchmark_d3qn.json`.
2. Claims are metric-backed.
3. No false "fully SOTA" claim.

---

## Time-Boxed Schedule

1. 09:00-09:30: Setup + baseline launch
2. 09:30-12:00: Vision validation + realtime integration proof
3. 12:00-15:00: RL benchmark run + result extraction
4. 15:00-17:00: Backend hardening quick pass
5. 17:00-18:30: Final summary artifacts + demo rehearsal

---

## Risk Controls (One-Day Reality)

1. If CUDA fails, continue on CPU and capture honest latency.
2. If long benchmark cannot finish, run lower timesteps and document scope.
3. If backend hardening causes regressions, gate by env flag and keep demo path stable.

---

## Definition of Done (End of Day)

You are done when all are true:

1. Dashboard demo runs reliably.
2. Vision evidence JSON exists and is presentable.
3. RL benchmark JSON exists and includes core traffic KPIs.
4. Basic backend hardening is active in non-demo mode.
5. One-day summary report is complete and honest.

---

## Next 7 Days (Post One-Day Sprint)

1. Replace basic tracker path with a stronger MOT pipeline.
2. Add repeatable test suite for endpoints and websocket stability.
3. Expand multi-scenario benchmarks and seed-robustness reporting.
4. Add proper auth/roles and signed control commands.
5. Start graph-temporal prediction upgrade beyond plain LSTM.
