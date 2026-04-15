# Repository Hygiene And Structure

This project has strong source separation already, but the root still mixes canonical code with generated artifacts, research outputs, and one-off assets. This note records what should be kept, what should remain ignored, and what needs review before any further restructuring.

## Keep

- `backend/main.py` as the current runtime backend entrypoint.
- `ai/rl/`, `ai/envs/`, `ai/utils/`, and `ai/explainability/` as the core RL and orchestration code.
- `ai/vision/`, `ai/prediction/`, `ai/anomaly/`, `control/`, and `iot/` as feature packages.
- `configs/`, `networks/`, `scripts/`, and `docs/` as project infrastructure.
- `train.py`, `evaluate.py`, `run_demo.py`, and `run_digital_twin.py` as public entrypoints.

## Generated Or Runtime Output

These should remain out of source review unless the specific artifact is being validated:

- `logs/`
- `models/`
- `results/`
- `reports/`
- `audio_cache/`
- `__pycache__/`

## Review Before Keeping In Git

- `render.yaml` and `requirements-deploy.txt` are deployment-specific and should only stay if deployment is part of the repo story.
- `presentation_assets/` and `presentation_diagrams/` are useful if they are curated deliverables; otherwise they should be reduced to the final screenshots/diagrams only.
- Root markdown notes such as `1.md` and other long scratch files should be archived or removed if they are not a deliberate documentation deliverable.
- Large generated PDFs, DOCX files, MP4s, and PNG/JPG screenshots should be kept only when they are final showcase artifacts.

## Recommended Target Structure

The current layout is close to the following clearer shape:

```text
.
├── backend/
├── ai/
├── data/
├── models/
├── frontend/
├── scripts/
├── reports/
├── tests/
├── config/
└── docs/
```

Suggested mapping from the existing tree:

- `dashboard/backend/` -> `backend/` (completed)
- `src/agents/`, `src/envs/`, `src/coordination/`, `src/utils/`, `prediction/`, and `vision/` -> `ai/` (completed)
- `iot/` -> `data/`
- `dashboard/frontend/` -> `frontend/` (completed)
- `configs/` -> `config/`
- curated benchmark/report outputs -> `reports/`

The backend/AI/frontend namespace migration is applied. Remaining optional cleanup is mostly organizational (for example `configs/` -> `config/`, `iot/` placement, and report artifact curation).

## Safe Cleanup Rules

1. Remove cache and temp folders first.
2. Keep only final showcase artifacts in the root.
3. Avoid moving core packages until the import graph is updated.
4. Prefer wrapper modules or deprecation shims over hard deletes for public entrypoints.