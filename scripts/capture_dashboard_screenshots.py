"""Capture dashboard screenshots for LinkedIn-ready sharing.

Usage:
  python scripts/capture_dashboard_screenshots.py
"""

from pathlib import Path
from playwright.sync_api import sync_playwright


OUT_DIR = Path("presentation_assets/linkedin")
BASE_URL = "http://127.0.0.1:8000"


def snap(page, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(OUT_DIR / name), full_page=True)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(3500)

        # 1) Hero authority dashboard shot
        snap(page, "01_authority_overview.png")

        # 2) Focused digital twin panel (cropped region via locator screenshot)
        page.locator("#cityCanvas").screenshot(path=str(OUT_DIR / "02_digital_twin_canvas.png"))

        # 3) Citizen portal
        page.get_by_role("button", name="Citizen Portal").click()
        page.wait_for_timeout(1200)
        snap(page, "03_citizen_portal.png")

        # 4) AI analytics
        page.get_by_role("button", name="AI Analytics").click()
        page.wait_for_timeout(1800)
        snap(page, "04_ai_analytics.png")

        # 5) System architecture
        page.get_by_role("button", name="System Architecture").click()
        page.wait_for_timeout(1200)
        snap(page, "05_system_architecture.png")

        browser.close()


if __name__ == "__main__":
    main()
