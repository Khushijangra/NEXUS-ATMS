"""Create a release-candidate manifest from benchmark artifacts.

This script validates gate pass, snapshots key KPIs, and records artifact hashes
so a promotion decision can be reproduced later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _build_manifest(
    gate_report_path: Path,
    summary_path: Path,
    output_path: Path,
    policy_name: str,
) -> Dict[str, Any]:
    gate = _load_json(gate_report_path)
    summary = _load_json(summary_path)

    if not bool(gate.get("gate_pass", False)):
        raise RuntimeError(
            "Gate did not pass. Use a passing gate artifact or rerun with valid policy."
        )

    aggregate = summary.get("aggregate", {}) if isinstance(summary, dict) else {}

    artifacts = {
        "gate_report": {
            "path": _rel(gate_report_path),
            "sha256": _sha256(gate_report_path),
        },
        "multiseed_summary": {
            "path": _rel(summary_path),
            "sha256": _sha256(summary_path),
        },
    }

    manifest: Dict[str, Any] = {
        "release_candidate": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy": policy_name,
            "status": "accepted",
            "timesteps": int(summary.get("timesteps", 0)),
            "config": summary.get("config"),
            "seeds": gate.get("seeds", []),
        },
        "gate": {
            "pass": True,
            "thresholds": gate.get("thresholds", {}),
            "counts": gate.get("counts", {}),
        },
        "kpi_snapshot": {
            "mean_reward": aggregate.get("mean_reward"),
            "std_reward": aggregate.get("std_reward"),
            "avg_waiting_time": aggregate.get("avg_waiting_time"),
            "avg_queue_length": aggregate.get("avg_queue_length"),
            "avg_throughput": aggregate.get("avg_throughput"),
            "any_nan_loss": aggregate.get("any_nan_loss"),
        },
        "artifacts": artifacts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize benchmark release candidate")
    parser.add_argument(
        "--gate-report",
        type=str,
        default="results/d3qn_gate_report_release.json",
        help="Path to a passing gate report JSON",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="results/d3qn_multiseed_summary.json",
        help="Path to multi-seed summary JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/release_candidate.json",
        help="Output manifest path",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="release-gate-15.25",
        help="Human-readable policy name",
    )
    args = parser.parse_args()

    gate_report_path = (ROOT / args.gate_report).resolve()
    summary_path = (ROOT / args.summary).resolve()
    output_path = (ROOT / args.output).resolve()

    manifest = _build_manifest(
        gate_report_path=gate_report_path,
        summary_path=summary_path,
        output_path=output_path,
        policy_name=args.policy,
    )

    print("[release] ACCEPTED")
    print(f"[release] output: {output_path}")
    print(f"[release] policy: {manifest['release_candidate']['policy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
