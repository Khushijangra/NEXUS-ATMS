"""
Vehicle Detector — OpenCV + YOLOv8 (Ultralytics)
===================================================
Supports three detection backends in order of preference:
  1. YOLOv8n (real-time, GPU-accelerated)          — requires ultralytics
  2. OpenCV DNN (MobileNet-SSD COCO, CPU fallback) — requires opencv-python
  3. Synthetic frame generator                     — no extra installs needed

Usage:
    detector = VehicleDetector(backend="yolo")   # or "dnn" / "synthetic"
    frame = cv2.imread("intersection.jpg")
    detections = detector.detect(frame)
    annotated  = detector.draw(frame, detections)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO vehicle class IDs (car, motorbike, bus, truck, bicycle)
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}
# Default confidence threshold
DEFAULT_CONF = 0.45
# Model weight paths (downloaded on first use by ultralytics)
YOLO_MODEL = "yolov8n.pt"


@dataclass
class Detection:
    """Single vehicle detection result."""
    track_id: int = -1             # assigned by tracker; -1 before tracking
    class_id: int = 2              # COCO class (default: car)
    label: str = "car"
    confidence: float = 1.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x1,y1,x2,y2 (pixels)
    center: Tuple[int, int] = (0, 0)
    timestamp: float = field(default_factory=time.time)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class VehicleDetector:
    """
    Multi-backend vehicle detector.

    Parameters
    ----------
    backend : str
        "yolo" | "dnn" | "synthetic"
    conf_threshold : float
        Minimum confidence to report a detection.
    frame_shape : tuple
        (H, W) used only in synthetic mode.
    """

    def __init__(
        self,
        backend: str = "yolo",
        conf_threshold: float = DEFAULT_CONF,
        frame_shape: Tuple[int, int] = (720, 1280),
        device: str = "cuda",
    ) -> None:
        self.backend = backend
        self.conf = conf_threshold
        self.frame_shape = frame_shape
        self.device = device

        self._model = None
        self._net = None
        self._classes: List[str] = []

        self._fps_counter = 0
        self._fps_t0 = time.time()
        self.fps: float = 0.0

        self._init_backend()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_backend(self) -> None:
        if self.backend == "yolo":
            self._init_yolo()
        elif self.backend == "dnn":
            self._init_dnn()
        else:
            logger.info("[Detector] Using synthetic backend (no camera required).")

    def _init_yolo(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(YOLO_MODEL)
            self._model.to(self.device)
            logger.info(f"[Detector] YOLOv8n loaded on {self.device}.")
        except ImportError:
            logger.warning(
                "[Detector] ultralytics not installed — falling back to DNN backend."
            )
            self.backend = "dnn"
            self._init_dnn()

    def _init_dnn(self) -> None:
        """Load MobileNet-SSD via OpenCV DNN (downloads weights once)."""
        try:
            import cv2  # type: ignore

            weights_dir = Path(__file__).parent / "weights"
            weights_dir.mkdir(exist_ok=True)

            proto = weights_dir / "MobileNetSSD_deploy.prototxt"
            model = weights_dir / "MobileNetSSD_deploy.caffemodel"

            if not proto.exists() or not model.exists():
                logger.info(
                    "[Detector] DNN weights missing — will use synthetic mode.\n"
                    "  Download MobileNetSSD prototxt + caffemodel into vision/weights/."
                )
                self.backend = "synthetic"
                return

            self._net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
            with open(weights_dir / "coco_labels.txt") as fh:
                self._classes = [l.strip() for l in fh]
            logger.info("[Detector] OpenCV DNN MobileNet-SSD loaded.")
        except ImportError:
            logger.warning("[Detector] opencv-python not installed — using synthetic.")
            self.backend = "synthetic"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: Optional[np.ndarray] = None) -> List[Detection]:
        """
        Run detection on *frame*.  If frame is None and backend == "synthetic",
        a synthetic frame is generated internally.

        Returns
        -------
        List[Detection]
        """
        self._fps_counter += 1
        if time.time() - self._fps_t0 >= 1.0:
            self.fps = self._fps_counter / (time.time() - self._fps_t0)
            self._fps_counter = 0
            self._fps_t0 = time.time()

        if self.backend == "yolo":
            return self._detect_yolo(frame)
        elif self.backend == "dnn":
            return self._detect_dnn(frame)
        else:
            return self._detect_synthetic()

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes + labels on *frame* (returns annotated copy)."""
        try:
            import cv2  # type: ignore
        except ImportError:
            return frame

        out = frame.copy()
        COLOR = (0, 200, 60)
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR, 2)
            label_txt = f"{d.label} {d.confidence:.2f}"
            cv2.putText(
                out, label_txt, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2,
            )
        cv2.putText(
            out, f"FPS: {self.fps:.1f}  Vehicles: {len(detections)}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2,
        )
        return out

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        results = self._model(frame, conf=self.conf, verbose=False)[0]
        detections: List[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=cls_id,
                    label=VEHICLE_CLASSES[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                )
            )
        return detections

    def _detect_dnn(self, frame: np.ndarray) -> List[Detection]:
        import cv2  # type: ignore

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        self._net.setInput(blob)
        detections_raw = self._net.forward()

        detections: List[Detection] = []
        for i in range(detections_raw.shape[2]):
            conf = float(detections_raw[0, 0, i, 2])
            cls_id = int(detections_raw[0, 0, i, 1])
            if conf < self.conf or cls_id not in VEHICLE_CLASSES:
                continue
            box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append(
                Detection(
                    class_id=cls_id,
                    label=VEHICLE_CLASSES[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                )
            )
        return detections

    def _detect_synthetic(self) -> List[Detection]:
        """Generate plausible fake detections for testing pipelines."""
        import random

        rng = random.Random()
        h, w = self.frame_shape
        n_vehicles = rng.randint(0, 12)
        detections: List[Detection] = []
        for i in range(n_vehicles):
            cls_id = rng.choice(list(VEHICLE_CLASSES.keys()))
            bw, bh = rng.randint(60, 120), rng.randint(40, 80)
            x1 = rng.randint(0, w - bw)
            y1 = rng.randint(0, h - bh)
            detections.append(
                Detection(
                    class_id=cls_id,
                    label=VEHICLE_CLASSES[cls_id],
                    confidence=round(rng.uniform(0.5, 0.99), 2),
                    bbox=(x1, y1, x1 + bw, y1 + bh),
                    center=(x1 + bw // 2, y1 + bh // 2),
                )
            )
        return detections
