"""
Anomaly Detector
=================
Real-time statistical anomaly detection for traffic sensor streams.
Uses an ensemble of three complementary detectors:

  1. **Z-score detector**         — flags values beyond μ ± k·σ
  2. **IQR detector**             — flags values outside 1.5×IQR fence
  3. **Rate-of-change detector**  — flags sudden jumps between consecutive steps

Both unsupervised (no labels needed) and label-calibration modes are supported.
Produces an AnomalyAlert when two or more detectors agree on the same feature.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyAlert:
    feature: str
    value: float
    expected_range: Tuple[float, float]
    detectors_fired: List[str]
    severity: str                       # "LOW" | "MEDIUM" | "HIGH"
    timestamp: float = field(default_factory=time.time)

    @property
    def message(self) -> str:
        lo, hi = self.expected_range
        return (
            f"Anomaly in '{self.feature}': {self.value:.2f} "
            f"(expected {lo:.2f}–{hi:.2f})  [{', '.join(self.detectors_fired)}]"
        )


class AnomalyDetector:
    """
    Rolling-window anomaly detector.

    Parameters
    ----------
    window      : int    Number of historical observations to keep per feature.
    z_threshold : float  Z-score cutoff (default 3.0 ≈ 0.3% false-positive rate).
    roc_factor  : float  Rate-of-change factor (flag if delta > factor × σ).
    min_samples : int    Minimum observations before triggering alerts.
    vote_quorum : int    Minimum number of detectors that must agree (1, 2, or 3).
    """

    def __init__(
        self,
        window: int = 120,
        z_threshold: float = 3.0,
        roc_factor: float = 4.0,
        min_samples: int = 30,
        vote_quorum: int = 2,
    ) -> None:
        self.window = window
        self.z_thresh = z_threshold
        self.roc_factor = roc_factor
        self.min_samples = min_samples
        self.vote_quorum = vote_quorum

        self._buffers: Dict[str, Deque[float]] = {}
        self._last_values: Dict[str, float] = {}
        self._alert_history: List[AnomalyAlert] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, features: Dict[str, float]) -> List[AnomalyAlert]:
        """
        Feed current values for named features.
        Returns list of new AnomalyAlerts (empty if nothing detected).
        """
        new_alerts: List[AnomalyAlert] = []

        for name, val in features.items():
            buf = self._buffers.setdefault(name, deque(maxlen=self.window))
            prev = self._last_values.get(name, val)

            alerts_fired: List[str] = []

            if len(buf) >= self.min_samples:
                arr = np.array(buf)
                mu, sig = arr.mean(), arr.std() + 1e-9

                # Z-score
                if abs(val - mu) / sig > self.z_thresh:
                    alerts_fired.append("z_score")

                # IQR
                q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
                iqr = q3 - q1
                if val < q1 - 1.5 * iqr or val > q3 + 1.5 * iqr:
                    alerts_fired.append("iqr")

                # Rate of change
                delta = abs(val - prev)
                if delta > self.roc_factor * sig:
                    alerts_fired.append("roc")

                if len(alerts_fired) >= self.vote_quorum:
                    lo = max(0.0, mu - self.z_thresh * sig)
                    hi = mu + self.z_thresh * sig
                    n_detectors = len(alerts_fired)
                    severity = "HIGH" if n_detectors == 3 else ("MEDIUM" if n_detectors == 2 else "LOW")
                    alert = AnomalyAlert(
                        feature=name,
                        value=val,
                        expected_range=(lo, hi),
                        detectors_fired=alerts_fired,
                        severity=severity,
                    )
                    new_alerts.append(alert)
                    self._alert_history.append(alert)
                    logger.warning(f"[Anomaly] {alert.message}")

            buf.append(val)
            self._last_values[name] = val

        return new_alerts

    def recent_alerts(self, n: int = 20) -> List[AnomalyAlert]:
        return list(reversed(self._alert_history[-n:]))

    def reset(self, feature: Optional[str] = None) -> None:
        if feature:
            self._buffers.pop(feature, None)
        else:
            self._buffers.clear()
            self._last_values.clear()

    def feature_stats(self) -> Dict[str, dict]:
        """Return running statistics for each feature."""
        out = {}
        for name, buf in self._buffers.items():
            if len(buf) < 2:
                continue
            arr = np.array(buf)
            out[name] = {
                "mean": round(float(arr.mean()), 3),
                "std": round(float(arr.std()), 3),
                "min": round(float(arr.min()), 3),
                "max": round(float(arr.max()), 3),
                "n": len(buf),
            }
        return out
