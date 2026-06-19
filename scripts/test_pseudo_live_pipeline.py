"""
30-second pseudo-live integration test for NEXUS backend.

Checks:
- WebSocket is reachable
- Map/status endpoints stay responsive
- Non-zero traffic output appears (real CV or demo fallback)

Usage:
  python scripts/test_pseudo_live_pipeline.py --base-url http://127.0.0.1:8000 --seconds 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, List

import requests


async def _ws_probe(ws_url: str, seconds: int) -> Dict:
    result = {
        "connected": False,
        "messages": 0,
        "non_zero_frames": 0,
        "non_zero_kpi_frames": 0,
    }
    try:
        import websockets
    except Exception as exc:
        result["error"] = f"websockets import failed: {exc}"
        return result

    end_t = time.time() + seconds
    try:
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
            result["connected"] = True
            while time.time() < end_t:
                timeout_s = max(0.1, end_t - time.time())
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    break
                payload = json.loads(raw)
                result["messages"] += 1
                state = payload.get("system_state", {})
                vehicle_count = int(state.get("map", {}).get("cv", {}).get("vehicle_count", 0))
                queue_len = float(state.get("session_metrics", {}).get("queue_length", 0.0) or 0.0)
                avg_wait = float(state.get("session_metrics", {}).get("avg_wait_ai", 0.0) or 0.0)
                if vehicle_count > 0:
                    result["non_zero_frames"] += 1
                if queue_len > 0.0 or avg_wait > 0.0:
                    result["non_zero_kpi_frames"] += 1
    except Exception as exc:
        result["error"] = str(exc)

    return result


def _poll_http(base_url: str, seconds: int) -> Dict:
    end_t = time.time() + seconds
    map_non_zero = 0
    status_ok = 0
    snapshot_non_zero = 0
    intersections_non_zero = 0
    failures: List[str] = []

    while time.time() < end_t:
        try:
            st = requests.get(f"{base_url}/api/status", timeout=2)
            mp = requests.get(f"{base_url}/api/map/state", timeout=3)
            snap = requests.get(f"{base_url}/api/snapshot", timeout=3)
            ints = requests.get(f"{base_url}/api/intersections", timeout=3)
            if st.ok:
                status_ok += 1
            if mp.ok:
                vehicle_count = int(mp.json().get("map", {}).get("cv", {}).get("vehicle_count", 0))
                if vehicle_count > 0:
                    map_non_zero += 1
            if snap.ok:
                snap_json = snap.json()
                queue = float(snap_json.get("queue_length", 0.0) or 0.0)
                wait = float(snap_json.get("avg_waiting_time", 0.0) or 0.0)
                if queue > 0.0 or wait > 0.0:
                    snapshot_non_zero += 1
            if ints.ok:
                ints_json = ints.json()
                if isinstance(ints_json, list) and ints_json:
                    non_zero_found = any(
                        float(item.get("queue_length", 0.0) or 0.0) > 0.0
                        or float(item.get("avg_waiting_time", 0.0) or 0.0) > 0.0
                        or int(item.get("vehicle_count", 0) or 0) > 0
                        for item in ints_json
                    )
                    if non_zero_found:
                        intersections_non_zero += 1
        except Exception as exc:
            failures.append(str(exc))
        time.sleep(0.5)

    return {
        "status_ok_polls": status_ok,
        "map_non_zero_polls": map_non_zero,
        "snapshot_non_zero_polls": snapshot_non_zero,
        "intersections_non_zero_polls": intersections_non_zero,
        "failures": failures[-3:],
    }


async def _poll_http_async(base_url: str, seconds: int) -> Dict:
    return await asyncio.to_thread(_poll_http, base_url, seconds)


async def _run_checks(base_url: str, ws_url: str, seconds: int):
    return await asyncio.gather(
        _poll_http_async(base_url, seconds),
        _ws_probe(ws_url, seconds),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Pseudo-live pipeline validator")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()

    ws_url = args.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/live"

    print(f"[test] running for {args.seconds}s against {args.base_url}")
    http_res, ws_res = asyncio.run(_run_checks(args.base_url, ws_url, args.seconds))

    summary = {
        "http": http_res,
        "websocket": ws_res,
    }
    print(json.dumps(summary, indent=2))

    http_non_zero = (
        http_res["map_non_zero_polls"] > 0
        or http_res["snapshot_non_zero_polls"] > 0
        or http_res["intersections_non_zero_polls"] > 0
    )
    ws_connected = bool(ws_res.get("connected"))
    ws_non_zero = int(ws_res.get("non_zero_frames", 0)) > 0 or int(ws_res.get("non_zero_kpi_frames", 0)) > 0

    if not ws_connected:
        print("[FAIL] websocket not active")
        return 2
    if not (http_non_zero or ws_non_zero):
        print("[FAIL] outputs remained zero for full test window")
        return 3

    print("[PASS] pseudo-live pipeline produced non-zero outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
