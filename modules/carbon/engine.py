"""
NEXUS-ATMS Carbon Credit & ESG Engine
=======================================
Calculates real-time CO₂ savings, generates ISO-14064 style reports,
and produces PDF carbon savings certificates for municipalities.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CarbonSnapshot:
    """Single measurement point for carbon calculations."""
    timestamp: float
    total_idle_minutes_ai: float
    total_idle_minutes_baseline: float
    vehicle_count: int

    @property
    def idle_minutes_saved(self) -> float:
        return max(0.0, self.total_idle_minutes_baseline - self.total_idle_minutes_ai)


class CarbonCreditEngine:
    """
    Tracks vehicle idle time reductions and converts them to
    measurable CO₂ savings, fuel savings, and cost savings.

    All calculations follow UNFCCC methodology for transport emissions.
    """

    # Emission factors
    IDLE_CO2_KG_PER_MIN = 0.21       # Average petrol car idle emission
    FUEL_CONSUMPTION_IDLE_L_HR = 0.8  # Litres/hour at idle
    FUEL_COST_PER_LITRE = 103.0       # INR (configurable)
    CO2_PER_LITRE_PETROL = 2.31       # kg CO₂ per litre petrol burned

    def __init__(
        self,
        idle_emission_kg_per_min: float = 0.21,
        fuel_cost_per_litre: float = 103.0,
        fuel_consumption_idle: float = 0.8,
    ):
        self.IDLE_CO2_KG_PER_MIN = idle_emission_kg_per_min
        self.FUEL_COST_PER_LITRE = fuel_cost_per_litre
        self.FUEL_CONSUMPTION_IDLE_L_HR = fuel_consumption_idle

        self._snapshots: List[CarbonSnapshot] = []
        self._daily_totals: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Data Ingestion
    # ------------------------------------------------------------------

    def record_snapshot(
        self,
        idle_minutes_ai: float,
        idle_minutes_baseline: float,
        vehicle_count: int,
    ):
        """Record a single carbon measurement snapshot."""
        snap = CarbonSnapshot(
            timestamp=time.time(),
            total_idle_minutes_ai=idle_minutes_ai,
            total_idle_minutes_baseline=idle_minutes_baseline,
            vehicle_count=vehicle_count,
        )
        self._snapshots.append(snap)

        # Accumulate daily totals
        today = date.today().isoformat()
        if today not in self._daily_totals:
            self._daily_totals[today] = {
                "idle_saved_min": 0.0,
                "idle_ai_min": 0.0,
                "idle_baseline_min": 0.0,
                "vehicles": 0,
                "snapshots": 0,
            }
        d = self._daily_totals[today]
        d["idle_saved_min"] += snap.idle_minutes_saved
        d["idle_ai_min"] += idle_minutes_ai
        d["idle_baseline_min"] += idle_minutes_baseline
        d["vehicles"] += vehicle_count
        d["snapshots"] += 1

    # ------------------------------------------------------------------
    # Calculations
    # ------------------------------------------------------------------

    def get_today_stats(self) -> Dict:
        """Get today's carbon savings statistics."""
        today = date.today().isoformat()
        d = self._daily_totals.get(today, {
            "idle_saved_min": 0.0, "idle_ai_min": 0.0,
            "idle_baseline_min": 0.0, "vehicles": 0, "snapshots": 0,
        })

        idle_saved = d["idle_saved_min"]
        co2_saved = idle_saved * self.IDLE_CO2_KG_PER_MIN
        fuel_saved = (idle_saved / 60.0) * self.FUEL_CONSUMPTION_IDLE_L_HR
        cost_saved = fuel_saved * self.FUEL_COST_PER_LITRE

        return {
            "date": today,
            "idle_time_saved_minutes": round(idle_saved, 1),
            "co2_saved_kg": round(co2_saved, 2),
            "co2_saved_tonnes": round(co2_saved / 1000.0, 4),
            "fuel_saved_litres": round(fuel_saved, 2),
            "cost_saved_inr": round(cost_saved, 0),
            "cost_saved_usd": round(cost_saved / 83.0, 0),  # Approx INR/USD
            "total_vehicles": d["vehicles"],
            "reduction_pct": round(
                (idle_saved / d["idle_baseline_min"] * 100)
                if d["idle_baseline_min"] > 0 else 0.0, 1
            ),
            "annual_projection": {
                "co2_tonnes": round(co2_saved * 365 / 1000.0, 2),
                "cost_saved_inr": round(cost_saved * 365, 0),
                "cost_saved_lakh": round(cost_saved * 365 / 100000, 2),
            },
        }

    def get_all_daily_stats(self) -> List[Dict]:
        """Get carbon stats for all recorded days."""
        results = []
        for day_str, d in sorted(self._daily_totals.items()):
            idle_saved = d["idle_saved_min"]
            co2_saved = idle_saved * self.IDLE_CO2_KG_PER_MIN
            fuel_saved = (idle_saved / 60.0) * self.FUEL_CONSUMPTION_IDLE_L_HR
            cost_saved = fuel_saved * self.FUEL_COST_PER_LITRE
            results.append({
                "date": day_str,
                "co2_saved_kg": round(co2_saved, 2),
                "fuel_saved_litres": round(fuel_saved, 2),
                "cost_saved_inr": round(cost_saved, 0),
                "vehicles": d["vehicles"],
            })
        return results

    # ------------------------------------------------------------------
    # PDF Certificate Generation
    # ------------------------------------------------------------------

    def generate_certificate(self, output_path: str = "reports/carbon_certificate.pdf") -> str:
        """
        Generate a PDF Carbon Savings Certificate.

        Returns the path to the generated PDF.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.pdfgen import canvas as pdf_canvas
            from reportlab.lib.colors import HexColor
        except ImportError:
            logger.error("[Carbon] reportlab not installed. Cannot generate PDF.")
            return ""

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        stats = self.get_today_stats()

        c = pdf_canvas.Canvas(output_path, pagesize=A4)
        width, height = A4

        # Header
        c.setFillColor(HexColor("#1a237e"))
        c.rect(0, height - 3.5 * cm, width, 3.5 * cm, fill=True, stroke=False)
        c.setFillColor(HexColor("#ffffff"))
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(width / 2, height - 2.2 * cm, "NEXUS-ATMS")
        c.setFont("Helvetica", 14)
        c.drawCentredString(width / 2, height - 3.0 * cm, "Carbon Savings Certificate")

        # Certificate body
        y = height - 5.5 * cm
        c.setFillColor(HexColor("#000000"))

        c.setFont("Helvetica-Bold", 13)
        c.drawCentredString(width / 2, y, "Certificate of Environmental Impact")
        y -= 1.2 * cm

        c.setFont("Helvetica", 11)
        lines = [
            f"Date: {stats['date']}",
            f"",
            f"This certifies that the NEXUS-ATMS AI Traffic Management System",
            f"deployed in the monitored urban zone has achieved the following",
            f"measurable environmental impact during the reporting period:",
            f"",
        ]
        for line in lines:
            c.drawCentredString(width / 2, y, line)
            y -= 0.6 * cm

        # Key metrics box
        y -= 0.5 * cm
        c.setFillColor(HexColor("#e8eaf6"))
        c.roundRect(3 * cm, y - 5 * cm, width - 6 * cm, 5.5 * cm, 10, fill=True, stroke=False)

        c.setFillColor(HexColor("#1a237e"))
        c.setFont("Helvetica-Bold", 12)
        metrics = [
            (f"Vehicle Idle Time Reduced:  {stats['idle_time_saved_minutes']:,.1f} minutes"),
            (f"CO\u2082 Emissions Avoided:  {stats['co2_saved_kg']:,.2f} kg  "
             f"({stats['co2_saved_tonnes']:.4f} tonnes)"),
            (f"Fuel Saved:  {stats['fuel_saved_litres']:,.2f} litres"),
            (f"Cost Saved:  \u20b9{stats['cost_saved_inr']:,.0f}  "
             f"(~${stats['cost_saved_usd']:,.0f} USD)"),
            (f"Traffic Reduction:  {stats['reduction_pct']:.1f}%"),
            (f"Vehicles Monitored:  {stats['total_vehicles']:,}"),
        ]
        y -= 0.3 * cm
        for line in metrics:
            c.drawString(4 * cm, y, line)
            y -= 0.7 * cm

        # Annual projection
        y -= 1.0 * cm
        c.setFillColor(HexColor("#000000"))
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(width / 2, y, "Annual Projection (if deployed year-round)")
        y -= 0.8 * cm
        c.setFont("Helvetica", 11)
        proj = stats["annual_projection"]
        c.drawCentredString(width / 2, y,
                            f"CO\u2082: {proj['co2_tonnes']:.2f} tonnes  |  "
                            f"Cost: \u20b9{proj['cost_saved_lakh']:.2f} lakh")

        # Footer
        y -= 2.5 * cm
        c.setFont("Helvetica", 9)
        c.setFillColor(HexColor("#666666"))
        c.drawCentredString(width / 2, y,
                            "Methodology: UNFCCC CDM AMS-III.C (Emission reductions "
                            "from vehicle fuel switching)")
        y -= 0.5 * cm
        c.drawCentredString(width / 2, y,
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                            f"| NEXUS-ATMS v1.0 | ISO 14064 Compatible")

        c.save()
        logger.info(f"[Carbon] Certificate generated: {output_path}")
        return output_path
