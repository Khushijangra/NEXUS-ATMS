import os
import torch
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
from pathlib import Path

class BehavioralScorer:
    """
    Computes a global behavioral anomaly score (Ab) for an intersection based on 
    the aggregated trajectory features of all currently tracked vehicles.
    """
    def __init__(self, model_path=None):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.is_fitted = False
        
        if model_path is not None and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_fitted = True
            
    def fit_dummy(self):
        """
        Fits a dummy distribution for the isolation forest if no pre-trained 
        model exists. This allows the system to run end-to-end immediately.
        Features: [mean_v, mean_a, max_a, mean_j, std_angle]
        """
        # Generate synthetic 'normal' driving data
        # Normal speed: 5-15 m/s, smooth accel/jerk, low lane deviation
        np.random.seed(42)
        normal_data = np.random.normal(loc=[10.0, 0.5, 1.5, 0.2, 0.05], 
                                       scale=[2.0, 0.2, 0.5, 0.1, 0.02], 
                                       size=(500, 5))
                                       
        self.model.fit(normal_data)
        self.is_fitted = True
        
    def save(self, model_path):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def score_trajectories(self, features: np.ndarray) -> torch.Tensor:
        """
        Takes (N, 5) trajectory features and returns a single Ab scalar tensor.
        Ab range is approximately [0, 1] where 1 is highly anomalous.
        """
        if not self.is_fitted:
            self.fit_dummy()
            
        # Isolation Forest returns 1 for inliers, -1 for outliers
        # decision_function returns average anomaly score, negative means anomaly
        scores = self.model.decision_function(features)
        
        # Invert and normalize to approximately [0, 1]
        # (Lower decision function = more anomalous)
        normalized_scores = 0.5 - (scores / 2.0)
        normalized_scores = np.clip(normalized_scores, 0, 1)
        
        # The global behavioral anomaly for the intersection is the max 
        # (or 90th percentile) of the individual vehicle anomalies.
        # If one vehicle is driving dangerously, the intersection is at risk.
        if len(normalized_scores) == 0:
            global_anomaly = 0.0
        else:
            global_anomaly = np.max(normalized_scores)
            
        # Format as SPGRL Ab tensor: (Batch=1, 1)
        Ab = torch.tensor([[global_anomaly]], dtype=torch.float32)
        
        return Ab
