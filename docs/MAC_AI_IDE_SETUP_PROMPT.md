# NEXUS-ATMS Mac Setup Prompt for Any AI IDE

Use this exact prompt in your AI IDE.

## Copy-Paste Prompt

You are my coding assistant. I am on macOS and want to run the GitHub repository end-to-end with minimum manual work.

Repository URL:
https://github.com/Khushijangra/NEXUS-ATMS

Your job:
1. Clone the repository if it is not already present.
2. Create and activate a Python virtual environment.
3. Install all dependencies from requirements.txt.
4. Detect and install missing system dependencies needed for this project on macOS.
5. Configure SUMO if missing, because simulation/training can require it.
6. Run the project in dashboard demo mode first to validate setup.
7. Validate that backend APIs and websocket are reachable.
8. Optionally run a short training/evaluation smoke test.
9. If any command fails, auto-diagnose and fix until successful.
10. At the end, print a final success report with exact commands I can reuse.

Rules:
- Do not ask me to perform technical steps unless strictly necessary.
- Prefer safe, non-destructive commands.
- Explain each major step in one sentence.
- Keep a running checklist and mark each step done or blocked.
- If a package fails on latest Python, try a compatible Python version (3.10 or 3.11).
- Do not stop after partial setup.

Implementation details to follow:

A) Environment setup
- Check for Homebrew. If missing, install it.
- Ensure these are installed: git, python, pip, and optional ffmpeg/opencv helpers if required by wheels.
- Create venv in project root named .venv.
- Activate venv and upgrade pip/setuptools/wheel.
- Install dependencies:
  pip install -r requirements.txt

B) SUMO setup (for simulation/training mode)
- Check if sumo command exists.
- If missing, install SUMO with Homebrew.
- Set SUMO_HOME for current shell and persist it in zsh profile.
- Verify with: sumo --version

C) Run dashboard first (recommended)
- Start command:
  python run_demo.py --dashboard-only
- If port 8000 is busy, automatically retry on a free port by setting uvicorn port env or backend launch arg.
- Verify endpoint health:
  GET http://127.0.0.1:8000/api/status
- Verify frontend opens:
  http://127.0.0.1:8000

D) Optional smoke test for training/eval
- Run a short train:
  python train.py --agent dqn --timesteps 5000 --demo
- Find produced best model under models/*/best/best_model.zip
- Evaluate:
  python evaluate.py --model <best_model_path> --agent dqn --report
- Confirm report exists in results/.

E) Troubleshooting policy
- If module import fails, install missing package in the active venv and re-run.
- If OpenCV or torch issues appear, try CPU-compatible variants first.
- If SUMO network or route files are missing, regenerate using scripts/generate_scenarios.py.
- If backend fails due to optional modules, run demo mode and continue.

F) Final output format required
- Print:
  1) What you installed
  2) What commands succeeded
  3) URLs that are live
  4) Any remaining non-blocking issues
  5) One reusable start command for daily use

Now execute all steps in order.

## Notes for the user

- The fastest first run is dashboard demo mode. It does not require full SUMO training pipeline to be healthy.
- Full simulation and training are more sensitive to local SUMO and package compatibility.
- If your AI IDE supports task automation, ask it to save successful commands as reusable tasks.
