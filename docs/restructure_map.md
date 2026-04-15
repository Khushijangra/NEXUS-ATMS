# Repository Restructure Map

This document records the practical migration from the legacy layout to the cleaner target layout for GitHub showcase and production-style review.

## What Was Cleaned In This Pass

- Removed `1.md` because it was a scratch note, not a source artifact.
- Removed the empty `nexus_dashboard/` folder.
- Removed temporary benchmark and validation folders from `logs/` and `models/` such as `benchmark_cpu_tmp`, `benchmark_eval_tmp`, `_sb3_device_check_*`, `_smoke_d3qn*`, `test_ckpt`, and `test_resume`.
- Promoted `backend/` and `ai/` to source-of-truth packages and updated runtime imports.
- Moved frontend source to `frontend/index.html`.
- Removed legacy implementation roots after compatibility validation.

## Before Vs After

### Current Layout

The repo now centers around these active source roots:

- `backend/`
- `ai/`
- `control/`
- `iot/`
- `modules/`
- `frontend/`
- `configs/`
- `networks/`
- `scripts/`
- `docs/`
- root entrypoints like `train.py`, `evaluate.py`, `run_demo.py`, and `run_digital_twin.py`

### Target Layout

```text
project-root/
├── backend/
│   ├── api/
│   ├── services/
│   ├── core/
│   └── main.py
├── ai/
│   ├── rl/
│   ├── prediction/
│   ├── anomaly/
│   └── vision/
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── models/
│   ├── trained/
│   └── checkpoints/
├── frontend/
├── scripts/
├── reports/
├── tests/
├── config/
└── docs/
```

## Exact Mapping

### Backend

- `dashboard/backend/main.py` -> `backend/main.py` (completed)
- `dashboard/backend/` support code -> `backend/api/`, `backend/services/`, and `backend/core/` (completed)
- `dashboard/frontend/index.html` -> `frontend/index.html` (completed)

### AI And Modeling

- `src/agents/` -> `ai/rl/` (completed)
- `src/coordination/` -> `ai/rl/` (completed)
- `prediction/` -> `ai/prediction/` and `ai/anomaly/` (completed)
- `vision/` -> `ai/vision/` (completed)
- `control/` remains as runtime control integration layer

### Data And Assets

- `iot/` -> `data/`
- `configs/` -> `config/`
- curated run outputs, benchmark snapshots, and final reports -> `reports/`

### Root Entry Points

- `train.py`
- `evaluate.py`
- `run_demo.py`
- `run_digital_twin.py`

These should remain as compatibility entrypoints until wrappers are added inside the new structure.

### Scripts

- `scripts/` stays as the operational and maintenance layer.
- Benchmarking, validation, report generation, and demo packaging scripts already belong here.

## Keep / Remove / Review

### Keep

- `backend/main.py`
- `ai/`
- `control/`
- `iot/`
- `scripts/`
- `configs/`
- `networks/`
- `frontend/index.html`
- `README.md`
- `requirements.txt`
- `requirements-deploy.txt`
- `render.yaml`

### Remove

- `1.md`
- Empty `nexus_dashboard/`
- Temporary logs and benchmark scratch folders in `logs/` and `models/`

### Review Before Keeping In Git

- `presentation_assets/`
- `presentation_diagrams/`
- `docs/video_frames/`
- `docs/*.mp4`, `docs/*.pdf`, `docs/*.docx` if they are not final showcase deliverables
- `results/` subfolders that are reproducible build artifacts rather than source of truth

## Migration State

The physical migration has been completed with compatibility verification:

1. Imports were rewritten to `backend.main` and `ai.*`.
2. Runtime backend startup was validated in demo-safe mode.
3. Legacy duplicate source roots were removed.
4. Remaining follow-up work is documentation alignment and optional repo organization cleanup.

## Practical Next Actions

1. Add wrapper entrypoints under `backend/` and `ai/`.
2. Consolidate any duplicate modules where overlap remains, especially anomaly detection and report generation.
3. Prune final showcase assets so only curated outputs remain in version control.
4. Add tests for the public CLI and backend health endpoints before any hard rename.