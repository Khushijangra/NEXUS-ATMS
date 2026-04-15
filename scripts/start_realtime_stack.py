"""Start dashboard backend and real-time detector with one command.

This script launches:
1) backend/main.py
2) scripts/run_realtime_detection.py

It streams logs from both processes and shuts both down on Ctrl+C.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _stream_output(prefix: str, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{prefix}] {line.rstrip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Start backend + realtime detection")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--backend", default="yolo", choices=["yolo", "dnn", "synthetic"])
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    backend_cmd = [PY, "backend/main.py"]
    detector_cmd = [
        PY,
        "scripts/run_realtime_detection.py",
        "--source",
        args.source,
        "--backend",
        args.backend,
        "--device",
        args.device,
    ]
    if args.no_display:
        detector_cmd.append("--no-display")
    if args.max_frames > 0:
        detector_cmd.extend(["--max-frames", str(args.max_frames)])

    print("=" * 60)
    print("NEXUS-ATMS realtime stack")
    print("Backend : http://127.0.0.1:8000")
    print(f"Detector: source={args.source} backend={args.backend} device={args.device}")
    print("=" * 60)

    backend = subprocess.Popen(
        backend_cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    t_backend = threading.Thread(target=_stream_output, args=("backend", backend), daemon=True)
    t_backend.start()

    # Give backend a moment to start before detector logs flood terminal.
    time.sleep(2)

    detector = subprocess.Popen(
        detector_cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    t_detector = threading.Thread(target=_stream_output, args=("detector", detector), daemon=True)
    t_detector.start()

    try:
        while True:
            if backend.poll() is not None:
                print("[stack] backend exited, stopping detector")
                break
            if detector.poll() is not None:
                print("[stack] detector exited, stopping backend")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[stack] Ctrl+C received, shutting down")

    for proc in (detector, backend):
        if proc.poll() is None:
            proc.terminate()
    time.sleep(0.5)
    for proc in (detector, backend):
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    main()
