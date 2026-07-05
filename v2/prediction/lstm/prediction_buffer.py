import numpy as np

class PredictionBuffer:
    def __init__(self, history_size=30, forecast_horizon=10, alpha=0.2):
        self.history_size = history_size
        self.forecast_horizon = forecast_horizon
        self.alpha = alpha  # EMA smoothing factor
        
        self.buffer = []
        self.smoothed_prediction = None
        self.smoothed_confidence = None
        self.smoothed_variance = None
        
    def add_prediction(self, timestamp, prediction, confidence, variance):
        """
        prediction: np.ndarray shape (forecast_horizon, feature_dim)
        confidence: np.ndarray shape (forecast_horizon,)
        variance: np.ndarray shape (forecast_horizon, feature_dim)
        """
        entry = {
            "timestamp": timestamp,
            "prediction": prediction,
            "confidence": confidence,
            "variance": variance,
            "horizon": self.forecast_horizon
        }
        
        self.buffer.append(entry)
        if len(self.buffer) > self.history_size:
            self.buffer.pop(0)
            
        self._update_ema(prediction, confidence, variance)
        
    def _update_ema(self, prediction, confidence, variance):
        if self.smoothed_prediction is None:
            self.smoothed_prediction = prediction
            self.smoothed_confidence = confidence
            self.smoothed_variance = variance
        else:
            self.smoothed_prediction = self.alpha * prediction + (1 - self.alpha) * self.smoothed_prediction
            self.smoothed_confidence = self.alpha * confidence + (1 - self.alpha) * self.smoothed_confidence
            self.smoothed_variance = self.alpha * variance + (1 - self.alpha) * self.smoothed_variance
            
    def get_latest_smoothed(self):
        return {
            "prediction": self.smoothed_prediction,
            "confidence": self.smoothed_confidence,
            "variance": self.smoothed_variance
        }
