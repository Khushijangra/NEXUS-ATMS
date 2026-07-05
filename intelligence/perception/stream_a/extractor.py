import numpy as np
import torch
import time
import logging
from typing import Dict, Any

from .loader import load_stream_a_models

logger = logging.getLogger(__name__)

class StreamAOnlineExtractor:
    """
    Runtime wrapper for the ARGUS Stream-A offline models.
    Takes 16-frame clips and outputs real-time anomaly severity.
    """
    def __init__(self, mulde_checkpoint_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing StreamAOnlineExtractor on {self.device}")
        
        # Load the models using the strict adapter
        VideoMAEFeatureExtractor, MULDEScorer = load_stream_a_models()
        
        # Instantiate VideoMAE (FP16 by default inside its init)
        logger.info("Loading VideoMAE (MCG-NJU/videomae-base)...")
        self.videomae = VideoMAEFeatureExtractor(device=self.device)
        
        # Instantiate MULDE from checkpoint
        logger.info(f"Loading MULDE checkpoint from {mulde_checkpoint_path}...")
        self.mulde = MULDEScorer.load_checkpoint(mulde_checkpoint_path, device=self.device)
        self.mulde.eval()

    def extract_anomaly(self, clip_array: np.ndarray) -> Dict[str, Any]:
        """
        Runs the full Stream-A inference pipeline on a 16-frame clip.
        clip_array: [16, 224, 224, 3] Numpy array (RGB, 0-255)
        """
        start_time = time.time()
        
        try:
            # 1. VideoMAE Feature Extraction
            # We wrap the clip in a list so the extractor treats it as a single "video"
            features = self.videomae.extract_from_frames([clip_array], batch_size=1)
            # features is shape (1, 768)
            features_tensor = torch.from_numpy(features).to(self.device)
            
            # 2. MULDE Anomaly Scoring
            with torch.no_grad():
                alpha_t = self.mulde.score_anomaly(features_tensor)
                
            alpha_val = float(alpha_t.cpu().numpy()[0])
            
            # We map NLL to a generic 0.0 - 1.0 severity. 
            # In a real deployment, this requires statistical thresholding.
            # Using an arbitrary sigmoid for now to simulate bounded severity.
            severity = 1.0 / (1.0 + np.exp(- (alpha_val - 5.0) / 2.0))
            
            latency_ms = (time.time() - start_time) * 1000.0
            
            return {
                "alpha_t": alpha_val,
                "severity": severity,
                "latency_ms": latency_ms,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"StreamA inference failed: {e}", exc_info=True)
            return {
                "alpha_t": 0.0,
                "severity": 0.0,
                "latency_ms": (time.time() - start_time) * 1000.0,
                "status": "error",
                "error": str(e)
            }
