"""Generate a concise markdown release note from benchmark promotion artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: Any, digits: int = 2) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _build_markdown(release_manifest: Dict[str, Any], gate_report: Dict[str, Any]) -> str:
    rc = release_manifest.get("release_candidate", {})
    gate = release_manifest.get("gate", {})
    kpi = release_manifest.get("kpi_snapshot", {})
    artifacts = release_manifest.get("artifacts", {})

    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    seed_rows: List[Dict[str, Any]] = gate_report.get("seed_results", [])

    lines: List[str] = []
    lines.append("# D3QN Release Note")
    lines.append("")
    lines.append(f"Generated: {created}")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- Status: {str(rc.get('status', 'unknown')).upper()}")
    lines.append(f"- Policy: {rc.get('policy', 'n/a')}")
    lines.append(f"- Gate pass: {gate.get('pass', False)}")
    lines.append(f"- Timesteps: {rc.get('timesteps', 'n/a')}")
    lines.append(f"- Config: {rc.get('config', 'n/a')}")
    lines.append(f"- Seeds: {', '.join(str(s) for s in rc.get('seeds', []))}")
    lines.append("")

    lines.append("## KPI Snapshot")
    lines.append(f"- Mean reward: {_fmt(kpi.get('mean_reward'), 4)}")
    lines.append(f"- Reward std: {_fmt(kpi.get('std_reward'), 4)}")
    lines.append(f"- Avg waiting time (s): {_fmt(kpi.get('avg_waiting_time'), 4)}")
    lines.append(f"- Avg queue length: {_fmt(kpi.get('avg_queue_length'), 4)}")
    lines.append(f"- Avg throughput: {_fmt(kpi.get('avg_throughput'), 4)}")
    lines.append(f"- Any NaN loss: {kpi.get('any_nan_loss', 'n/a')}")
    lines.append("")

    lines.append("## Gate Thresholds")
    thresholds = gate.get("thresholds", {})
    counts = gate.get("counts", {})
    lines.append(f"- Required improved seeds: {thresholds.get('require_improved_seeds', 'n/a')}")
    lines.append(f"- Max throughput drop pct: {thresholds.get('max_throughput_drop_pct', 'n/a')}")
    lines.append(f"- Improved wait seeds: {counts.get('improved_wait', 'n/a')}")
    lines.append(f"- Improved queue seeds: {counts.get('improved_queue', 'n/a')}")
    lines.append(f"- Improved both seeds: {counts.get('improved_both', 'n/a')}")
    lines.append(f"- Acceptable throughput seeds: {counts.get('acceptable_throughput', 'n/a')}")
    lines.append("")

    lines.append("## Per-Seed Summary")
    lines.append("| Seed | Wait Change % | Queue Change % | Throughput Change % |")
    lines.append("|---:|---:|---:|---:|")
    for row in seed_rows:
        lines.append(
            "| "
            + f"{row.get('seed', 'n/a')}"
            + " | "
            + f"{_fmt(row.get('wait_change_pct'), 3)}"
            + " | "
            + f"{_fmt(row.get('queue_change_pct'), 3)}"
            + " | "
            + f"{_fmt(row.get('throughput_change_pct'), 3)}"
            + " |"
        )
    lines.append("")

    lines.append("## Artifact Integrity")
    gate_art = artifacts.get("gate_report", {})
    sum_art = artifacts.get("multiseed_summary", {})
    lines.append(f"- Gate report: {gate_art.get('path', 'n/a')}")
    lines.append(f"- Gate report sha256: {gate_art.get('sha256', 'n/a')}")
    lines.append(f"- Summary: {sum_art.get('path', 'n/a')}")
    lines.append(f"- Summary sha256: {sum_art.get('sha256', 'n/a')}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate markdown release notes")
    parser.add_argument(
        "--release-manifest",
        type=str,
        default="results/release_candidate.json",
        help="Path to locked release manifest JSON",
    )
    parser.add_argument(
        "--gate-report",
        type=str,
        default="results/d3qn_gate_report_release.json",
        help="Path to gate report JSON used for release",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/release_notes.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    release_manifest_path = (ROOT / args.release_manifest).resolve()
    gate_report_path = (ROOT / args.gate_report).resolve()
    output_path = (ROOT / args.output).resolve()

    release_manifest = _load_json(release_manifest_path)
    gate_report = _load_json(gate_report_path)

    markdown = _build_markdown(release_manifest, gate_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"[release-notes] output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
