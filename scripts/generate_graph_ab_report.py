"""Generate Graph OFF vs Graph ON A/B report from benchmark output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pct_improve(lower_is_better: bool, baseline: float, contender: float) -> float:
    if baseline == 0.0:
        return 0.0
    if lower_is_better:
        return ((baseline - contender) / abs(baseline)) * 100.0
    return ((contender - baseline) / abs(baseline)) * 100.0


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _build_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    envs = payload.get("environments", {})
    multi = envs.get("multi_agent", {}) if isinstance(envs, dict) else {}

    baseline = multi.get("d3qn", {}) if isinstance(multi, dict) else {}
    graph = multi.get("graph_d3qn", {}) if isinstance(multi, dict) else {}

    b_metrics = baseline.get("metrics", {}) if isinstance(baseline, dict) else {}
    g_metrics = graph.get("metrics", {}) if isinstance(graph, dict) else {}

    has_baseline = bool(b_metrics)
    has_graph = bool(g_metrics)

    if not has_baseline or not has_graph:
        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "insufficient_data",
            "message": "Benchmark file must include both multi_agent.d3qn and multi_agent.graph_d3qn metrics.",
            "has_baseline": has_baseline,
            "has_graph": has_graph,
        }

    comp = {
        "waiting_time_improve_pct": _pct_improve(True, float(b_metrics.get("avg_waiting_time", 0.0)), float(g_metrics.get("avg_waiting_time", 0.0))),
        "queue_length_improve_pct": _pct_improve(True, float(b_metrics.get("avg_queue_length", 0.0)), float(g_metrics.get("avg_queue_length", 0.0))),
        "reward_improve_pct": _pct_improve(False, float(b_metrics.get("mean_reward", 0.0)), float(g_metrics.get("mean_reward", 0.0))),
        "stability_improve_pct": _pct_improve(False, float(b_metrics.get("stability", 0.0)), float(g_metrics.get("stability", 0.0))),
        "spillback_improve_pct": _pct_improve(True, float(b_metrics.get("spillback_rate", 0.0)), float(g_metrics.get("spillback_rate", 0.0))),
    }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "meta": payload.get("meta", {}),
        "baseline": b_metrics,
        "graph": g_metrics,
        "comparison": comp,
        "recommendation": {
            "promote_graph": bool(
                comp["waiting_time_improve_pct"] >= 0.0
                and comp["queue_length_improve_pct"] >= 0.0
                and comp["spillback_improve_pct"] >= 0.0
            )
        },
    }


def _to_markdown(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Graph-D3QN A/B Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    if report.get("status") != "ok":
        lines.append("## Status")
        lines.append(f"- status: {report.get('status')}")
        lines.append(f"- message: {report.get('message')}")
        lines.append(f"- has_baseline: {report.get('has_baseline')}")
        lines.append(f"- has_graph: {report.get('has_graph')}")
        lines.append("")
        return "\n".join(lines)

    baseline = report.get("baseline", {})
    graph = report.get("graph", {})
    comp = report.get("comparison", {})
    rec = report.get("recommendation", {})

    lines.append("## Core Metrics")
    lines.append("| Metric | D3QN Baseline | Graph-D3QN | Improvement % |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Waiting Time | {_fmt(baseline.get('avg_waiting_time'))} | {_fmt(graph.get('avg_waiting_time'))} | {_fmt(comp.get('waiting_time_improve_pct'), 2)} |"
    )
    lines.append(
        f"| Queue Length | {_fmt(baseline.get('avg_queue_length'))} | {_fmt(graph.get('avg_queue_length'))} | {_fmt(comp.get('queue_length_improve_pct'), 2)} |"
    )
    lines.append(
        f"| Mean Reward | {_fmt(baseline.get('mean_reward'))} | {_fmt(graph.get('mean_reward'))} | {_fmt(comp.get('reward_improve_pct'), 2)} |"
    )
    lines.append(
        f"| Stability | {_fmt(baseline.get('stability'))} | {_fmt(graph.get('stability'))} | {_fmt(comp.get('stability_improve_pct'), 2)} |"
    )
    lines.append(
        f"| Spillback Rate | {_fmt(baseline.get('spillback_rate'))} | {_fmt(graph.get('spillback_rate'))} | {_fmt(comp.get('spillback_improve_pct'), 2)} |"
    )
    lines.append("")

    lines.append("## Recommendation")
    lines.append(f"- promote_graph: {bool(rec.get('promote_graph', False))}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Graph-D3QN A/B report")
    parser.add_argument("--benchmark", default="results/benchmark_d3qn.json", help="Input benchmark JSON")
    parser.add_argument("--json-out", default="results/graph_ab_report.json", help="Output report JSON")
    parser.add_argument("--md-out", default="results/graph_ab_report.md", help="Output report markdown")
    args = parser.parse_args()

    benchmark_path = (ROOT / args.benchmark).resolve()
    json_out_path = (ROOT / args.json_out).resolve()
    md_out_path = (ROOT / args.md_out).resolve()

    payload = _load_json(benchmark_path)
    report = _build_report(payload)
    markdown = _to_markdown(report)

    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    with json_out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_out_path.parent.mkdir(parents=True, exist_ok=True)
    with md_out_path.open("w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"[graph-ab] json: {json_out_path}")
    print(f"[graph-ab] markdown: {md_out_path}")
    print(f"[graph-ab] status: {report.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
