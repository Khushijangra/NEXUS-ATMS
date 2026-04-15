"""
LSTM Traffic Flow Predictor
============================
Predicts short-term traffic flow (vehicle counts, queue lengths, occupancy)
for each intersection approach up to *horizon* steps ahead.

Architecture
------------
  Input  : [batch, seq_len, n_features]  — sliding window of sensor readings
  Encoder: 2-layer bidirectional LSTM
  Decoder: 1-layer LSTM  (multi-step ahead)
  Output : [batch, horizon, n_approaches × n_targets]

Features per time step (per approach):
  vehicle_count, occupancy_pct, speed_kmh, queue_length,
  hour_sin, hour_cos, day_of_week_sin, day_of_week_cos,
  rainfall_norm, emergency_flag

Multiple step outputs:
  5-min, 10-min, 15-min, 30-min ahead (configurable)

Training data can be generated from the SensorSimulator for offline
pre-training, then fine-tuned on real deployment data.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Graceful degradation when PyTorch is not installed
_TORCH_OK = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    logger.warning(
        "[LSTM] PyTorch not found. LSTMPredictor will use statistical fallback.\n"
        "  Install: pip install torch"
    )


# -----------------------------------------------------------------------
# PyTorch model definition
# -----------------------------------------------------------------------

if _TORCH_OK:
    class _EncoderDecoder(nn.Module):
        """Seq2Seq LSTM encoder-decoder for multi-step traffic forecasting."""

        def __init__(
            self,
            n_features: int,
            n_outputs: int,
            horizon: int,
            hidden: int = 128,
            n_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.horizon = horizon
            self.hidden = hidden

            # Encoder
            self.encoder = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
                bidirectional=True,
            )
            # Bridge (bi→uni)
            self.bridge = nn.Linear(hidden * 2, hidden)

            # Decoder
            self.decoder_cell = nn.LSTMCell(input_size=n_outputs, hidden_size=hidden)
            self.out_proj = nn.Linear(hidden, n_outputs)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            x: "torch.Tensor",
            teacher_forcing_ratio: float = 0.0,
            target: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            # x : [B, T, F]
            enc_out, (h_n, c_n) = self.encoder(x)
            # Merge bidirectional hidden: take last layer's forward+backward
            h = torch.tanh(self.bridge(
                torch.cat([h_n[-2], h_n[-1]], dim=-1)
            ))  # [B, hidden]
            c = torch.zeros_like(h)

            # Seed decoder with last encoder output
            last_enc = enc_out[:, -1, :]
            dec_in = self.out_proj(h)           # [B, n_outputs]

            outputs = []
            for t in range(self.horizon):
                h, c = self.decoder_cell(dec_in, (h, c))
                pred = self.out_proj(self.dropout(h))   # [B, n_outputs]
                outputs.append(pred.unsqueeze(1))
                # Teacher forcing during training
                if (
                    target is not None
                    and torch.rand(1).item() < teacher_forcing_ratio
                ):
                    dec_in = target[:, t, :]
                else:
                    dec_in = pred

            return torch.cat(outputs, dim=1)  # [B, horizon, n_outputs]


# -----------------------------------------------------------------------
# Predictor wrapper
# -----------------------------------------------------------------------

N_APPROACHES = 4
# Features per approach per timestep
APPROACH_FEATURES = 4   # count, occupancy, speed, queue
# Global features per timestep
GLOBAL_FEATURES = 6     # hour_sin, hour_cos, dow_sin, dow_cos, rain, emergency


class LSTMPredictor:
    """
    High-level interface for training and inference.

    Parameters
    ----------
    horizon      : int    Number of future steps to predict.
    seq_len      : int    Look-back window length (steps).
    step_minutes : float  Real time per step (minutes) — 5 min default.
    hidden       : int    LSTM hidden units.
    device       : str    "cuda" | "cpu".
    model_path   : str    Where to save/load weights.
    """

    def __init__(
        self,
        horizon: int = 6,           # 6 × 5 min = 30-min ahead
        seq_len: int = 24,          # 24 × 5 min = 2 h look-back
        step_minutes: float = 5.0,
        hidden: int = 128,
        device: str = "cuda",
        model_path: str = "models/lstm_predictor.pt",
    ) -> None:
        self.horizon = horizon
        self.seq_len = seq_len
        self.step_minutes = step_minutes
        self.hidden = hidden
        # Force GPU if available, fallback to CPU
        if device == "cuda" and _TORCH_OK and torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.empty_cache()
            logger.info(f"[LSTM] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("[LSTM] Using CPU")
        self.model_path = model_path

        n_feat = N_APPROACHES * APPROACH_FEATURES + GLOBAL_FEATURES
        n_out  = N_APPROACHES * APPROACH_FEATURES

        self._model = None
        self._fitted = False
        self._rolling_buffer: List[np.ndarray] = []   # raw feature vectors

        if _TORCH_OK:
            self._model = _EncoderDecoder(
                n_features=n_feat,
                n_outputs=n_out,
                horizon=horizon,
                hidden=hidden,
            ).to(device)
            # Load pre-trained weights if available
            if os.path.isfile(model_path):
                self._model.load_state_dict(
                    torch.load(model_path, map_location=device)
                )
                self._fitted = True
                logger.info(f"[LSTM] Loaded weights from {model_path}.")
            else:
                logger.info("[LSTM] No saved weights — model needs training.")

        # Statistical fallback buffer (moving average per feature)
        self._stat_buffer: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_observation(self, snapshot) -> None:
        """
        Enqueue one IntersectionSnapshot into the rolling buffer.
        Call this every sensing interval (e.g. every 5 minutes).
        """
        vec = self._snapshot_to_vector(snapshot)
        self._rolling_buffer.append(vec)
        if len(self._rolling_buffer) > self.seq_len:
            self._rolling_buffer.pop(0)

    def predict(self) -> Optional[np.ndarray]:
        """
        Return predicted feature vectors for the next *horizon* steps.

        Returns
        -------
        np.ndarray  shape [horizon, n_outputs]  or None if buffer too short.
        """
        if len(self._rolling_buffer) < self.seq_len:
            return None

        if _TORCH_OK and self._fitted and self._model is not None:
            return self._predict_nn()
        else:
            return self._predict_statistical()

    def train(
        self,
        snapshots,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> List[float]:
        """
        Train on a list of IntersectionSnapshots.
        Returns list of epoch losses.
        """
        if not _TORCH_OK:
            logger.warning("[LSTM] PyTorch unavailable — training skipped.")
            return []

        vecs = [self._snapshot_to_vector(s) for s in snapshots]
        X, y = self._build_dataset(np.array(vecs))
        if X is None:
            return []

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
        loss_fn = nn.HuberLoss()

        losses: List[float] = []
        self._model.train()
        for ep in range(epochs):
            ep_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                pred = self._model(xb, teacher_forcing_ratio=0.5, target=yb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()
            ep_loss /= len(loader)
            losses.append(ep_loss)
            sched.step(ep_loss)
            if ep % 10 == 0 or ep == epochs - 1:
                logger.info(f"[LSTM] Epoch {ep+1}/{epochs}  loss={ep_loss:.4f}")

        # Save weights
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), self.model_path)
        self._fitted = True
        logger.info(f"[LSTM] Model saved to {self.model_path}.")
        return losses

    def forecast_horizon_labels(self) -> List[str]:
        """Human-readable labels for each forecast step."""
        return [
            f"+{int((i + 1) * self.step_minutes)} min"
            for i in range(self.horizon)
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _predict_nn(self) -> np.ndarray:
        self._model.eval()
        x = np.array(self._rolling_buffer[-self.seq_len:])
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._model(xt)   # [1, horizon, n_out]
        return out.squeeze(0).cpu().numpy()

    def _predict_statistical(self) -> np.ndarray:
        """Simple moving-average fallback (no PyTorch needed)."""
        buf = np.array(self._rolling_buffer[-self.seq_len:])
        n_out_feats = N_APPROACHES * APPROACH_FEATURES
        ma = buf[:, :n_out_feats].mean(axis=0)
        # Repeat for each horizon step with small decay
        result = np.array([
            ma * max(0.85, 1.0 - 0.02 * h)
            for h in range(self.horizon)
        ])
        return result

    def _snapshot_to_vector(self, snapshot) -> np.ndarray:
        """Convert IntersectionSnapshot to flat feature vector."""
        APPROACHES = ("north", "south", "east", "west")
        ap_feats = []
        for ap_name in APPROACHES:
            ap = getattr(snapshot, "approaches", {}).get(ap_name, None)
            if ap is not None:
                ap_feats += [
                    ap.vehicle_count / 20.0,
                    ap.occupancy_pct / 100.0,
                    ap.speed_kmh / 80.0,
                    ap.queue_length / 30.0,
                ]
            else:
                ap_feats += [0.0, 0.0, 0.0, 0.0]

        # Temporal features
        import datetime
        now = datetime.datetime.now()
        h = now.hour + now.minute / 60.0
        dow = now.weekday()
        global_feats = [
            math.sin(2 * math.pi * h / 24),
            math.cos(2 * math.pi * h / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            min(getattr(snapshot, "rainfall_mm_h", 0.0), 50.0) / 50.0,
            float(getattr(snapshot, "emergency_active", False)),
        ]
        return np.array(ap_feats + global_feats, dtype=np.float32)

    def _build_dataset(
        self, vecs: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        T = len(vecs)
        n_out = N_APPROACHES * APPROACH_FEATURES
        if T < self.seq_len + self.horizon:
            logger.warning("[LSTM] Not enough data to build training dataset.")
            return None, None
        X, y = [], []
        for i in range(T - self.seq_len - self.horizon + 1):
            X.append(vecs[i: i + self.seq_len])
            y.append(vecs[i + self.seq_len: i + self.seq_len + self.horizon, :n_out])
        return np.array(X), np.array(y)
