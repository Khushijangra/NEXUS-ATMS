import numpy as np

def compute_kinematic_features(track_history, fps=30.0):
    """
    Computes kinematics (velocity, acceleration, jerk, path length) from 
    a history of detections.
    """
    if len(track_history) < 4:
        # Not enough history to compute up to jerk
        return np.zeros(5, dtype=np.float32)
        
    dt = 1.0 / fps
    
    # Extract centers (x, y)
    centers = np.array([det.center for det in track_history]) # Shape: (N, 2)
    
    # Position differences (dx, dy)
    dp = np.diff(centers, axis=0)
    
    # Velocity vectors
    v = dp / dt
    v_mag = np.linalg.norm(v, axis=1)
    
    # Acceleration vectors
    dv = np.diff(v, axis=0)
    a = dv / dt
    a_mag = np.linalg.norm(a, axis=1)
    
    # Jerk vectors
    da = np.diff(a, axis=0)
    j = da / dt
    j_mag = np.linalg.norm(j, axis=1)
    
    # Aggregated features
    mean_v = np.mean(v_mag)
    mean_a = np.mean(a_mag)
    max_a = np.max(a_mag)
    mean_j = np.mean(j_mag)
    
    # Lane deviation (proxy: standard deviation of lateral movement or angle)
    # Simple proxy: variance in the trajectory heading
    if len(v) > 1:
        angles = np.arctan2(v[:, 1], v[:, 0])
        std_angle = np.std(angles)
    else:
        std_angle = 0.0
        
    features = np.array([mean_v, mean_a, max_a, mean_j, std_angle], dtype=np.float32)
    
    # Sanitize NaNs
    features = np.nan_to_num(features, nan=0.0)
    return features
    
class TrajectoryFeatureExtractor:
    def __init__(self, fps=30.0, max_history=30):
        self.fps = fps
        self.max_history = max_history
        
    def extract(self, tracker):
        """
        Takes a DeepSORTTracker and returns a (N, F) feature matrix
        for all active tracks with sufficient history.
        """
        feature_list = []
        for track in tracker.tracks:
            # Only process tracks that have settled
            if track.hits >= 4:
                history = track.history[-self.max_history:]
                feats = compute_kinematic_features(history, self.fps)
                feature_list.append(feats)
                
        if len(feature_list) == 0:
            return np.zeros((1, 5), dtype=np.float32)
            
        return np.vstack(feature_list)
