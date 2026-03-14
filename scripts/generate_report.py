"""
Report Generator for Smart Traffic Management System
Generates HTML + PNG visual reports from evaluation results.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import (
    plot_comparison_bar,
    plot_learning_curve,
    generate_report_figure,
)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Smart Traffic System — Evaluation Report</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px auto; max-width: 900px; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
  th {{ background: #3498db; color: white; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .improvement {{ color: #27ae60; font-weight: bold; }}
  .regression {{ color: #e74c3c; font-weight: bold; }}
  .chart {{ text-align: center; margin: 30px 0; }}
  .chart img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  footer {{ text-align: center; margin-top: 40px; color: #888; font-size: 13px; }}
</style>
</head>
<body>
<h1>🚦 Smart Traffic Management — Performance Report</h1>
<p>Generated: {timestamp}</p>

<h2>Performance Comparison</h2>
<table>
<tr><th>Metric</th><th>Fixed-Timing</th><th>RL Agent</th><th>Change</th></tr>
{rows}
</table>

<div class="chart">
<h2>Visual Comparison</h2>
<img src="comparison_chart.png" alt="Comparison Chart">
</div>

<footer>AI-Based Smart Traffic Management System &copy; {year}</footer>
</body>
</html>"""


def _format_change(pct: float, metric: str) -> str:
    # For throughput, positive is good; for others, negative is good
    good = pct > 0 if metric == "throughput" else pct < 0
    cls = "improvement" if good else "regression"
    return f'<span class="{cls}">{pct:+.1f}%</span>'


def generate_html_report(results_path: str, output_dir: str) -> str:
    """Create an HTML report from evaluation_results.json."""
    with open(results_path) as f:
        data = json.load(f)

    comparison = data.get("comparison", {})
    rows = ""
    for metric, vals in comparison.items():
        label = metric.replace("_", " ").title()
        change_html = _format_change(vals["change_pct"], metric)
        rows += (
            f"<tr><td>{label}</td>"
            f"<td>{vals['baseline']:.2f}</td>"
            f"<td>{vals['rl_agent']:.2f}</td>"
            f"<td>{change_html}</td></tr>\n"
        )

    html = HTML_TEMPLATE.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        year=datetime.now().year,
        rows=rows,
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        "--results",
        type=str,
        default="results/evaluation_results.json",
        help="Path to evaluation results JSON",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        print("Run evaluate.py first to generate results.")
        sys.exit(1)

    # Generate chart
    with open(args.results) as f:
        data = json.load(f)

    chart_path = os.path.join(args.output_dir, "comparison_chart.png")
    plot_comparison_bar(
        data.get("baseline", {}),
        data.get("rl_agent", {}),
        save_path=chart_path,
    )
    print(f"[OK] Chart → {chart_path}")

    # Generate HTML
    html_path = generate_html_report(args.results, args.output_dir)
    print(f"[OK] Report → {html_path}")


if __name__ == "__main__":
    main()
