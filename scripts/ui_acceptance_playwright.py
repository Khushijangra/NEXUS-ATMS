from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


BASE_URL = "http://127.0.0.1:8000"


def _click_expect_post(page, selector: str, endpoint_fragment: str, timeout_ms: int = 12000) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        with page.expect_response(
            lambda r: endpoint_fragment in r.url and r.request.method == "POST",
            timeout=timeout_ms,
        ) as response_info:
            page.click(selector)
        resp = response_info.value
        status = int(resp.status)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "ok": 200 <= status < 300,
            "selector": selector,
            "endpoint": endpoint_fragment,
            "status": status,
            "latency_ms": round(dt_ms, 1),
        }
    except Exception as exc:
        return {"ok": False, "selector": selector, "endpoint": endpoint_fragment, "error": str(exc)}


def run_ui_acceptance() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ws_connected": False,
        "ui_checks": [],
        "action_checks": [],
        "demo_sequence": {},
        "pill_snapshot": "",
        "pill_source": "",
        "overall_pass": False,
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)

        # Wait for websocket-driven status label.
        try:
            page.wait_for_function(
                "() => (document.getElementById('wsStatus')?.textContent || '').toLowerCase().includes('connected')",
                timeout=18000,
            )
            result["ws_connected"] = True
        except PlaywrightTimeoutError:
            result["ws_connected"] = False

        # Board mode toggles should activate corresponding tab state.
        mode_checks: List[Dict[str, Any]] = []
        for selector, mode in [
            ("#boardModeHeat", "heat"),
            ("#boardModeFlow", "flow"),
            ("#boardModeRanked", "ranked"),
        ]:
            try:
                page.click(selector)
                active = page.eval_on_selector(selector, "el => el.classList.contains('active')")
                mode_checks.append({"mode": mode, "ok": bool(active)})
            except Exception as exc:
                mode_checks.append({"mode": mode, "ok": False, "error": str(exc)})
        result["ui_checks"].extend(mode_checks)

        # Demo sequence should trigger a chain of backend actions.
        hits: Dict[str, int] = {
            "/api/mode/set": 0,
            "/api/signal/override": 0,
            "/api/emergency/activate": 0,
            "/api/security/simulate": 0,
            "/api/voice/announce": 0,
        }
        statuses: Dict[str, List[int]] = {k: [] for k in hits}

        def _resp_listener(resp):
            if resp.request.method != "POST":
                return
            for k in hits:
                if k in resp.url:
                    hits[k] += 1
                    statuses[k].append(int(resp.status))

        page.on("response", _resp_listener)
        try:
            page.click("#adminDemoRun")
            page.wait_for_timeout(8500)
        finally:
            try:
                page.remove_listener("response", _resp_listener)
            except Exception:
                pass

        toast_text = ""
        try:
            toast_text = (page.locator("#toast").inner_text(timeout=1500) or "").strip().lower()
        except Exception:
            toast_text = ""

        result["demo_sequence"] = {
            "hits": hits,
            "statuses": statuses,
            "toast": toast_text,
            "ok": all(v >= 1 for v in hits.values()) and all((statuses[k] and all(200 <= s < 300 for s in statuses[k])) for k in statuses),
        }

        # Core control strip + admin action buttons.
        action_checks: List[Dict[str, Any]] = []
        action_checks.append(_click_expect_post(page, "#btnReturnAi", "/api/mode/set"))
        action_checks.append(_click_expect_post(page, "#btnNS", "/api/signal/override"))
        action_checks.append(_click_expect_post(page, "#btnEW", "/api/signal/override"))
        action_checks.append(_click_expect_post(page, "#btnAllRed", "/api/signal/override"))
        action_checks.append(_click_expect_post(page, "#adminEmergency", "/api/emergency/activate"))
        action_checks.append(_click_expect_post(page, "#adminSimAttack", "/api/security/simulate"))
        action_checks.append(_click_expect_post(page, "#adminVoice", "/api/voice/announce"))
        result["action_checks"] = action_checks

        # Snapshot/source truth pills should be present and populated.
        try:
            result["pill_snapshot"] = (page.locator("#pillSnapshot").inner_text(timeout=5000) or "").strip()
        except Exception:
            result["pill_snapshot"] = ""
        try:
            result["pill_source"] = (page.locator("#pillSourceTruth").inner_text(timeout=5000) or "").strip()
        except Exception:
            result["pill_source"] = ""

        browser.close()

    ui_ok = all(bool(x.get("ok")) for x in result["ui_checks"])
    action_ok = all(bool(x.get("ok")) for x in result["action_checks"])
    pills_ok = bool(result["pill_snapshot"]) and bool(result["pill_source"])
    result["overall_pass"] = bool(result["ws_connected"] and ui_ok and action_ok and result["demo_sequence"].get("ok") and pills_ok)
    return result


if __name__ == "__main__":
    out = run_ui_acceptance()
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out.get("overall_pass") else 1)
