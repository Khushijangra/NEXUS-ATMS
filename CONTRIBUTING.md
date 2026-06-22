# Contributing to NEXUS-ATMS

Thanks for contributing.

## Development setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start dashboard backend:

```bash
python dashboard/backend/main.py
```

## Branching

- Create feature branches from `main`.
- Use branch names like:
  - `feat/<short-name>`
  - `fix/<short-name>`
  - `docs/<short-name>`

## Commit style

Use concise, scoped messages:

- `feat: add live camera frame ingestion`
- `fix: handle missing SUMO_HOME gracefully`
- `docs: update setup instructions`

## Pull request checklist

Before opening a PR, ensure:

- [ ] Code runs locally without syntax errors.
- [ ] Relevant scripts or endpoints were tested.
- [ ] README/docs updated if behavior changed.
- [ ] No secrets, local env files, or large generated artifacts were added.

## Testing guidance

Recommended checks before PR:

```bash
python -m compileall -q src control iot prediction vision modules dashboard scripts train.py evaluate.py run_demo.py run_digital_twin.py
```

## Reporting issues

Use GitHub Issues and include:

- Exact command used
- Full error trace
- OS + Python version
- Whether SUMO is installed and configured

## Scope notes

The repository currently supports both:

- Demo mode (synthetic/live-like dashboard)
- Simulation-backed workflows (SUMO scripts)

When contributing, clearly mark whether your change targets demo mode, SUMO mode, or both.
