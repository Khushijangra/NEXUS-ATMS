from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "nonzero_demo_tables_v4"
OUTPUT_PDF = ROOT / "docs" / "nonzero_demo_tables_report.pdf"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_table(rows: list[dict], headers: list[str], col_widths: list[float], font_size: int = 8) -> Table:
    data = [headers]
    for row in rows:
        data.append([row.get(h, "") for h in headers])
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b5c7d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f9fd")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def main() -> int:
    wide_rows = read_csv(INPUT_DIR / "junction_waiting_times_nonzero_wide.csv")
    corridor_rows = read_csv(INPUT_DIR / "green_corridor_events_nonzero.csv")
    summary_text = (INPUT_DIR / "nonzero_summary.txt").read_text(encoding="utf-8")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CenterTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CenterSub",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.HexColor("#444444"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SectionHdr",
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
        fontSize=9.2,
        leading=12,
        spaceAfter=6,
    ))

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=landscape(A4),
        rightMargin=28,
        leftMargin=28,
        topMargin=28,
        bottomMargin=24,
        title="NEXUS-ATMS Nonzero Traffic Tables",
        author="NEXUS-ATMS",
    )

    story = []
    story.append(Paragraph("NEXUS-ATMS Nonzero Traffic Tables", styles["CenterTitle"]))
    story.append(Paragraph("Waiting Time at All Junctions and Green Corridor Data", styles["CenterSub"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Time-Series Waiting Time Table", styles["SectionHdr"]))
    wide_headers = ["Time", "Tick", "J0", "J1", "J2", "J3", "AvgWait", "Phase", "Mode"]
    story.append(build_table(wide_rows[:24], wide_headers, [1.55 * inch, 0.55 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 0.80 * inch, 0.95 * inch, 0.55 * inch], font_size=7.5))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Green Corridor Event Table", styles["SectionHdr"]))
    corridor_headers = ["Time", "Tick", "Junction", "Action", "DelayBefore_s", "DelayAfter_s", "VehicleType", "VehicleId", "EventId"]
    story.append(build_table(corridor_rows[:20], corridor_headers, [1.45 * inch, 0.45 * inch, 0.55 * inch, 1.15 * inch, 0.80 * inch, 0.80 * inch, 0.90 * inch, 0.75 * inch, 1.20 * inch], font_size=7.3))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Interpretation", styles["SectionHdr"]))
    story.append(Paragraph(
        "The waiting-time series shows nonzero variation across time and across the aggregated junction groups J0, J1, J2, and J3. "
        "The corridor table records the emergency corridor activation path and the corresponding delay reduction before versus after the override. "
        "These outputs are appropriate for a paper section reporting time-series behavior and emergency green-corridor response.",
        styles["BodySmall"],
    ))

    def draw_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(doc.leftMargin, 18, "NEXUS-ATMS Nonzero Traffic Tables")
        canvas.drawRightString(landscape(A4)[0] - doc.rightMargin, 18, f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(f"Created {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
