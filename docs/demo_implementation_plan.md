# Demo Implementation Plan (End-to-End)

## Objective
Deliver a stable, eye-catching project demo video with verified backend/frontend connectivity, working admin controls, and reproducible evidence artifacts.

## Current Implementation Status
- SUMO and TraCI connection: verified.
- Dashboard backend: running successfully.
- Admin panel in frontend: implemented and connected.
- Admin actions validated:
  - Security simulate: working.
  - Emergency corridor activation: working.
  - Audit logs feed: working.

## Phase 1 - Environment Bring-Up
1. Use Python 3.13 runtime.
2. Validate SUMO:
   - `python scripts/test_sumo_connection.py`
3. Start backend:
   - `python backend/main.py`
4. Open dashboard:
   - `http://127.0.0.1:8000`

Pass criteria:
- `/api/status` returns `status=running`
- `modules_loaded` is 8/8
- `runtime.frame_ok=true`

## Phase 2 - Frontend and Admin Verification
In the dashboard:
1. Check WS status is connected.
2. Verify frame/tick updates.
3. Use Admin Panel:
   - Activate Emergency Corridor
   - Simulate Security Event
   - Broadcast voice message
   - Refresh panel
4. Confirm audit feed updates in Admin Panel.

Pass criteria:
- No visible UI errors.
- Admin actions produce visible state/audit change.

## Phase 3 - Evidence Artifacts
1. Run evaluation with selected model:
   - `python evaluate.py --model <best_model_path> --agent d3qn --report`
2. Generate report:
   - `python scripts/generate_report.py`
3. Build demo submission bundle:
   - `python scripts/prepare_demo_submission.py`
4. Generate final report doc:
   - `python scripts/generate_dti_final_report.py`

Expected outputs:
- `results/evaluation_results.json`
- `results/comparison_chart.png`
- `results/report.html`
- `results/demo_submission_*/manifest.json`
- `results/DTI_Project_Final_Report_NEXUS_ATMS.docx`

## Phase 4 - Recording Strategy (3-5 min)
1. 20s: Problem framing.
2. 40s: Architecture overview.
3. 120s: Live command center + Admin Panel actions.
4. 60s: Metrics and comparison chart.
5. 20s: Closing impact statement.

## Fallback Matrix
- Plan A: full live dashboard + admin actions.
- Plan B: recorded UI playback + live report walkthrough.
- Plan C: static evidence only (report + chart + bundle manifest).

## Demo-Day Checklist
- [ ] Backend starts cleanly
- [ ] WS connected
- [ ] Admin emergency action works
- [ ] Admin security simulation works
- [ ] Audit feed shows latest events
- [ ] Evaluation artifacts present
- [ ] Demo bundle generated
- [ ] Final report ready
- [ ] One uninterrupted recording take complete
