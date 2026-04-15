"""
Prepare a submission-ready demo bundle from existing training/evaluation artifacts.

This script avoids retraining. It collects the latest valid artifacts, generates
training plots from D3QN history when available, and writes a concise demo brief.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.utils.visualization import plot_reward, plot_epsilon, plot_loss  # noqa: E402


def _latest_dir(parent: Path, prefix: str) -> Optional[Path]:
    if not parent.exists():
        return None
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _find_latest_model_file(model_root: Path) -> Optional[Path]:
    if not model_root.exists():
        return None

    candidate_files = []
    for run_dir in model_root.iterdir():
        if not run_dir.is_dir():
            continue

        best_d3qn = run_dir / "best" / "best_model.pt"
        best_sb3 = run_dir / "best" / "best_model.zip"
        final_d3qn = run_dir / "d3qn_final.pt"
        final_sb3_dqn = run_dir / "dqn_final.zip"
        final_sb3_ppo = run_dir / "ppo_final.zip"

        for candidate in [best_d3qn, best_sb3, final_d3qn, final_sb3_dqn, final_sb3_ppo]:
            if candidate.exists():
                candidate_files.append(candidate)

    if not candidate_files:
        return None

    return sorted(candidate_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _locate_history(logs_root: Path, explicit_log_dir: Optional[str]) -> Optional[Path]:
    if explicit_log_dir:
        candidate = ROOT / explicit_log_dir / "d3qn_history.json"
        if candidate.exists():
            return candidate
        return None

    latest_d3qn_log = _latest_dir(logs_root, "d3qn_")
    if latest_d3qn_log is None:
        return None

    candidate = latest_d3qn_log / "d3qn_history.json"
    if candidate.exists():
        return candidate
    return None


def _generate_history_plots(history_path: Path, out_plots_dir: Path) -> Dict[str, Optional[str]]:
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    out_plots_dir.mkdir(parents=True, exist_ok=True)
    reward_path = plot_reward(history, save_dir=str(out_plots_dir))
    epsilon_path = plot_epsilon(history, save_dir=str(out_plots_dir))
    loss_path = plot_loss(history, save_dir=str(out_plots_dir))

    return {
        "reward": reward_path,
        "epsilon": epsilon_path,
        "loss": loss_path,
    }


def _build_demo_brief(bundle_dir: Path, manifest: Dict) -> Path:
    brief_path = bundle_dir / "DEMO_BRIEF.md"

    lines = [
        "# Demo Video Brief",
        "",
        "## Video Outline (3-5 min)",
        "1. Problem statement (15-25 sec)",
        "2. Architecture snapshot (40-60 sec)",
        "3. Baseline vs RL visual comparison (60-90 sec)",
        "4. Results and key metrics (45-75 sec)",
        "5. Reliability note: checkpoint + resume (20-30 sec)",
        "",
        "## Submission Artifacts Included",
        f"- Model file: {manifest.get('model_file', 'N/A')}",
        f"- Evaluation JSON: {manifest.get('evaluation_results', 'N/A')}",
        f"- Comparison chart: {manifest.get('comparison_chart', 'N/A')}",
        f"- HTML report: {manifest.get('html_report', 'N/A')}",
        f"- Reward plot: {manifest.get('reward_plot', 'N/A')}",
        f"- Epsilon plot: {manifest.get('epsilon_plot', 'N/A')}",
        f"- Loss plot: {manifest.get('loss_plot', 'N/A')}",
        "",
        "## Presenter Notes",
        "- Avoid live retraining in the final recording.",
        "- Use precomputed artifacts from this bundle.",
        "- Keep claims tied to visible metrics only.",
    ]

    brief_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return brief_path


def prepare_bundle(log_dir: Optional[str], output_dir: Optional[str]) -> Tuple[Path, Dict]:
    logs_root = ROOT / "logs"
    results_root = ROOT / "results"
    models_root = ROOT / "models"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = ROOT / (output_dir or f"results/demo_submission_{timestamp}")
    bundle_dir.mkdir(parents=True, exist_ok=True)

    assets_dir = bundle_dir / "assets"
    plots_dir = assets_dir / "plots"
    assets_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_dir": str(bundle_dir.relative_to(ROOT)),
    }

    model_file = _find_latest_model_file(models_root)
    if model_file is not None:
        dst = assets_dir / "model" / model_file.name
        _copy_if_exists(model_file, dst)
        manifest["model_file"] = str(dst.relative_to(ROOT))
    else:
        manifest["model_file"] = None

    eval_json_src = results_root / "evaluation_results.json"
    eval_chart_src = results_root / "comparison_chart.png"
    eval_html_src = results_root / "report.html"

    eval_json_dst = assets_dir / "evaluation" / "evaluation_results.json"
    eval_chart_dst = assets_dir / "evaluation" / "comparison_chart.png"
    eval_html_dst = assets_dir / "evaluation" / "report.html"

    if _copy_if_exists(eval_json_src, eval_json_dst):
        manifest["evaluation_results"] = str(eval_json_dst.relative_to(ROOT))
    else:
        manifest["evaluation_results"] = None

    if _copy_if_exists(eval_chart_src, eval_chart_dst):
        manifest["comparison_chart"] = str(eval_chart_dst.relative_to(ROOT))
    else:
        manifest["comparison_chart"] = None

    if _copy_if_exists(eval_html_src, eval_html_dst):
        manifest["html_report"] = str(eval_html_dst.relative_to(ROOT))
    else:
        manifest["html_report"] = None

    history_path = _locate_history(logs_root, log_dir)
    if history_path is not None:
        history_dst = assets_dir / "training" / "d3qn_history.json"
        _copy_if_exists(history_path, history_dst)
        manifest["history_json"] = str(history_dst.relative_to(ROOT))

        plot_paths = _generate_history_plots(history_path, plots_dir)
        manifest["reward_plot"] = (
            str(Path(plot_paths["reward"]).relative_to(ROOT)) if plot_paths["reward"] else None
        )
        manifest["epsilon_plot"] = (
            str(Path(plot_paths["epsilon"]).relative_to(ROOT)) if plot_paths["epsilon"] else None
        )
        manifest["loss_plot"] = (
            str(Path(plot_paths["loss"]).relative_to(ROOT)) if plot_paths["loss"] else None
        )
    else:
        manifest["history_json"] = None
        manifest["reward_plot"] = None
        manifest["epsilon_plot"] = None
        manifest["loss_plot"] = None

    brief_path = _build_demo_brief(bundle_dir, manifest)
    manifest["demo_brief"] = str(brief_path.relative_to(ROOT))

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return bundle_dir, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare submission-ready demo bundle")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Optional log directory under project root (e.g., logs/d3qn_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory under project root",
    )
    args = parser.parse_args()

    bundle_dir, manifest = prepare_bundle(log_dir=args.log_dir, output_dir=args.output_dir)

    print("[OK] Demo submission bundle ready")
    print(f"[OK] Bundle: {bundle_dir}")
    print(f"[OK] Manifest: {bundle_dir / 'manifest.json'}")

    if manifest.get("history_json"):
        print(f"[OK] History plots: {bundle_dir / 'assets' / 'plots'}")
    else:
        print("[WARN] D3QN history not found. Skipped reward/epsilon/loss plots.")

    if not manifest.get("evaluation_results"):
        print("[WARN] evaluation_results.json not found in results/. Run evaluate.py for comparison metrics.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
