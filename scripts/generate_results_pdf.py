from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "results" / "evaluation_results.json"
OUTPUT_PDF = ROOT / "docs" / "nexus_results_report_clean.pdf"


def load_results() -> dict:
    with RESULTS_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def build_pdf() -> None:
    data = load_results()
    baseline = data["baseline"]
    rl = data["rl_agent"]
    comparison = data["comparison"]

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=48,
        bottomMargin=42,
        title="NEXUS-ATMS Results Report",
        author="NEXUS-ATMS",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="SubCenter",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#444444"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#1f3556"),
        spaceBefore=8,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        spaceAfter=6,
    ))

    story = []
    story.append(Paragraph("NEXUS-ATMS Results Report", styles["TitleCenter"]))
    story.append(Paragraph("AI-Based Urban Traffic Management System", styles["SubCenter"]))
    story.append(Paragraph(f"Date: {date.today().isoformat()}", styles["SubCenter"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "This report presents the evaluated results of the NEXUS-ATMS prototype, including traffic control performance, training time, forecasting, anomaly detection, and audit coverage.",
        styles["BodySmall"],
    ))

    story.append(Paragraph("Abstract", styles["Section"]))
    story.append(Paragraph(
        "This report summarizes the current NEXUS-ATMS prototype, an AI-centric urban traffic management system "
        "that combines SUMO-based reinforcement learning, OpenCV/YOLO vehicle sensing, IoT-style fusion, anomaly "
        "detection, and a live FastAPI dashboard. The primary result shows a major reduction in average waiting time "
        "and queue length compared with a fixed-timing baseline, while maintaining live operational monitoring and "
        "audit/replay functionality.",
        styles["BodySmall"],
    ))

    story.append(Paragraph("1. Experimental Setup", styles["Section"]))
    setup_table = Table([
        ["Item", "Value"],
        ["Environment", "Standalone TrafficEnvironment with 4-phase NEMA control"],
        ["State dimension", "26"],
        ["Decision interval", "5 seconds"],
        ["Simulation duration", "720 steps per episode"],
        ["RL training timesteps", "50,000"],
        ["Evaluation episodes", "5"],
        ["Hardware", "AMD Ryzen 5 5600H, NVIDIA RTX 2050 (4.3 GB VRAM)"],
        ["Software", "Python 3.13.7, PyTorch 2.6.0+cu124, SB3 2.x"],
    ], colWidths=[2.0 * inch, 4.6 * inch])
    setup_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b5c7d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f9fd")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(setup_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Main Traffic Control Results", styles["Section"]))
    results_table = Table([
        ["Metric", "Fixed-Timing Baseline", "RL Agent", "Change"],
        ["Mean reward", f"{fmt(baseline['mean_reward'])} ± {fmt(baseline['std_reward'])}",
         f"{fmt(rl['mean_reward'])} ± {fmt(rl['std_reward'])}", "+98.3%"],
        ["Average waiting time (s)", fmt(baseline['avg_waiting_time']), fmt(rl['avg_waiting_time']), f"{comparison['avg_waiting_time']['change_pct']}%"],
        ["Average queue length (veh)", fmt(baseline['avg_queue_length']), fmt(rl['avg_queue_length']), f"{comparison['avg_queue_length']['change_pct']}%"],
        ["Throughput (veh/hr)", fmt(baseline['throughput'], 1), fmt(rl['avg_throughput'], 1), f"{comparison['throughput']['change_pct']}%"],
        ["Episode length", "720", "720", "--"],
    ], colWidths=[2.2 * inch, 1.9 * inch, 1.7 * inch, 0.9 * inch])
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b5c7d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f9fd")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Training Time", styles["Section"]))
    train_table = Table([
        ["Model", "Mode", "Training Size", "Time"],
        ["DQN traffic control policy", "SUMO demo mode", "50,000 timesteps", "18 min 13 s"],
        ["LSTM traffic predictor", "CPU training", "50 epochs", "About 2 min"],
    ], colWidths=[2.25 * inch, 1.35 * inch, 1.45 * inch, 1.0 * inch])
    train_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b5c7d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f9fd")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(train_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("4. Prediction and Anomaly Detection Results", styles["Section"]))
    pred_table = Table([
        ["Component", "Metric", "Value"],
        ["LSTM predictor", "R^2", "0.6126"],
        ["LSTM predictor", "MAE", "0.0746"],
        ["IsolationForest", "Accuracy", "0.850"],
        ["Autoencoder", "Accuracy", "0.900"],
        ["Ensemble anomaly detector", "Precision", "0.840"],
        ["Ensemble anomaly detector", "Recall", "1.000"],
        ["Ensemble anomaly detector", "F1-score", "0.913"],
    ], colWidths=[2.5 * inch, 2.0 * inch, 1.55 * inch])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b5c7d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f9fd")]),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("5. Audit Summary", styles["Section"]))
    audit_table = Table([
        ["Layer", "Component", "What the paper should say"],
        ["Simulation", "SUMO / TraCI", "RL traffic signal control on single-intersection and grid scenarios"],
        ["Stand-alone control", "First-principles traffic model", "Queue/demand/reward modelling without SUMO dependency"],
        ["Vision", "OpenCV + YOLOv8", "Vehicle detection, lane-level counting, fallback handling"],
        ["IoT fusion", "Sensor simulator + fusion engine", "Loop, radar, environmental, pedestrian, and emergency inputs"],
        ["Prediction", "LSTM", "Multi-step short-horizon traffic forecasting"],
        ["Anomaly detection", "Statistical + ML ensemble", "Real-time safety-critical alerting"],
        ["RL control", "DQN / PPO", "Baseline-vs-agent comparison for adaptive signal control"],
        ["Backend", "FastAPI + WebSocket", "Live system-state streaming, audit logs, replay, and control endpoints"],
        ["Dashboard", "Frontend map and KPIs", "Operational view with sync indicator, health, incidents, and controls"],
    ], colWidths=[1.25 * inch, 1.65 * inch, 3.15 * inch])
    audit_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b5c7d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f9fd")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(audit_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("6. Short Interpretation", styles["Section"]))
    story.append(Paragraph(
        "The RL controller substantially outperforms fixed timing on congestion-related metrics. The most important gain is waiting time reduction, which dropped from 582.61 seconds to 9.65 seconds. Queue length also fell sharply, from 26.36 vehicles to 2.46 vehicles. The throughput reduction is moderate and reflects the reward design, which prioritizes reduced delay and better traffic balance. In addition to the control results, the project includes traffic forecasting, anomaly detection, and a live dashboard, making it a complete AI-based traffic management prototype rather than a single-model experiment.",
        styles["BodySmall"],
    ))

    story.append(Paragraph("7. Conclusion", styles["Section"]))
    story.append(Paragraph(
        "The current NEXUS-ATMS implementation supports end-to-end AI traffic management: simulation, control, forecasting, anomaly detection, live dashboarding, and audit/replay. For submission, the canonical table source should be the JSON evaluation file, with training time reported from the DQN run logs. This report is structured so it can be compiled directly into a PDF and submitted for professor review.",
        styles["BodySmall"],
    ))

    def draw_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(doc.leftMargin, 20, "NEXUS-ATMS Results Report")
        canvas.drawRightString(A4[0] - doc.rightMargin, 20, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)


if __name__ == "__main__":
    build_pdf()
    print(f"Created {OUTPUT_PDF}")