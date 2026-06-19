"""Real-time vehicle detection demo (YOLO/DNN/synthetic).

Examples:
    python scripts/run_realtime_detection.py --source 0 --backend yolo --device cpu
  python scripts/run_realtime_detection.py --source videos/traffic.mp4 --backend yolo
  python scripts/run_realtime_detection.py --backend synthetic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Union
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from ai.vision.counter import ZoneCounter
from ai.vision.detector import Detection, VehicleDetector
from ai.vision.speed_estimator import SpeedEstimator
from ai.vision.tracker import VehicleTracker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("realtime-detection")


def parse_source(src: str) -> Union[int, str]:
    return int(src) if src.isdigit() else src


def draw_hud(
    frame: np.ndarray,
    tracked: list[Detection],
    zone_stats: dict,
    avg_speed: float,
    backend: str,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    y = 28
    cv2.putText(out, f"Backend: {backend.upper()}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y += 28
    cv2.putText(out, f"Tracked vehicles: {len(tracked)}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 28
    cv2.putText(out, f"Avg speed: {avg_speed:.1f} km/h", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

    # Compact per-zone queue/throughput summary
    line = " | ".join(
        f"{name[0].upper()}:Q{vals['queue']}/T{vals['throughput']}"
        for name, vals in zone_stats.items()
    )
    cv2.putText(out, line, (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

    # Small top-right watermark
    cv2.putText(out, "NEXUS-ATMS Live Detection", (w - 330, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-time vehicle detection")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or video path")
    parser.add_argument("--backend", default="yolo", choices=["yolo", "dnn", "synthetic"])
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--save", default="", help="Optional output video path (e.g. results/live_detect.mp4)")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = infinite)")
    parser.add_argument(
        "--no-source-fallback",
        action="store_true",
        help="Fail instead of falling back to synthetic mode when source cannot open",
    )
    args = parser.parse_args()

    detector = VehicleDetector(
        backend=args.backend,
        conf_threshold=args.conf,
        frame_shape=(args.height, args.width),
        device=args.device,
    )
    tracker = VehicleTracker(iou_threshold=0.35, max_age=8, min_hits=1)
    counter = ZoneCounter(frame_shape=(args.height, args.width))
    speed = SpeedEstimator(fps=25.0)

    cap = None
    if args.backend != "synthetic":
        source = parse_source(args.source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            if args.no_source_fallback:
                raise RuntimeError(f"Could not open source: {args.source}")
            logger.warning(
                "Could not open source '%s'. Falling back to synthetic backend.",
                args.source,
            )
            if cap is not None:
                cap.release()
            cap = None
            detector = VehicleDetector(
                backend="synthetic",
                conf_threshold=args.conf,
                frame_shape=(args.height, args.width),
                device=args.device,
            )

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 25.0, (args.width, args.height))

    print("=" * 60)
    print("NEXUS-ATMS Real-Time Vehicle Detection")
    print(f"backend={detector.backend} device={detector.device} source={args.source}")
    if args.no_display:
        print("Running in headless mode")
    else:
        print("Press 'q' to quit")
    print("=" * 60)

    last_ts = time.time()
    frame_count = 0
    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (args.width, args.height))
        else:
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        detections = detector.detect(frame)
        tracked = tracker.update(detections)

        # Ensure counter sees frame-level objects even if tracker drops some.
        zone_stats = counter.update(tracked if tracked else detections)
        avg_speed = speed.average_speed(tracker.active_tracks())

        annotated = detector.draw(frame, tracked if tracked else detections)
        annotated = draw_hud(
            annotated,
            tracked if tracked else detections,
            zone_stats,
            avg_speed,
            detector.backend,
        )

        now = time.time()
        dt = max(now - last_ts, 1e-6)
        last_ts = now
        cv2.putText(annotated, f"Loop FPS: {1.0 / dt:.1f}", (12, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 60), 2)

        if writer is not None:
            writer.write(annotated)

        frame_count += 1

        if not args.no_display:
            cv2.imshow("NEXUS-ATMS Live Vehicle Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
