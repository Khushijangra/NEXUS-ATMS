import logging
import torch
import numpy as np

from .provider import FrameProvider, FrameBuffer
from .extractor import StreamAOnlineExtractor

logger = logging.getLogger(__name__)

class ARGUSEngine:
    """
    Singleton orchestrator for the ARGUS Flow perception pipeline.
    Connects a FrameProvider to the StreamAOnlineExtractor and caches anomaly state.
    """
    def __init__(self, frame_provider: FrameProvider, mulde_checkpoint: str, device: str = None):
        self.provider = frame_provider
        self.buffer = FrameBuffer(clip_length=16, stride=4)
        
        # Instantiate extractor
        self.extractor = StreamAOnlineExtractor(mulde_checkpoint_path=mulde_checkpoint, device=device)
        
        self._current_severity = 0.0
        self._current_alpha = 0.0
        
    def warmup(self):
        """Pre-allocates CUDA memory and compiles graphs using a dummy clip."""
        logger.info("Warming up ARGUSEngine...")
        dummy_clip = np.zeros((16, 224, 224, 3), dtype=np.uint8)
        self.extractor.extract_anomaly(dummy_clip)
        logger.info("ARGUSEngine warmup complete.")
        
    def step(self):
        """
        Advances the perception pipeline by one frame.
        Should be called synchronously during the simulation step.
        """
        frame = self.provider.get_frame()
        if frame is None:
            # Reached end of video or source failed, hold previous severity
            return
            
        self.buffer.push(frame)
        
        if self.buffer.is_ready():
            clip = self.buffer.get_clip()
            result = self.extractor.extract_anomaly(clip)
            
            if result["status"] == "success":
                self._current_severity = result["severity"]
                self._current_alpha = result["alpha_t"]
                
    def get_current_anomaly(self) -> float:
        """Returns the cached anomaly severity (0.0 to 1.0) for the RL environment."""
        return self._current_severity
        
    def get_current_alpha(self) -> float:
        """Returns the raw alpha(t) NLL for research plotting."""
        return self._current_alpha
        
    def shutdown(self):
        """Releases heavy PyTorch resources."""
        logger.info("Shutting down ARGUSEngine...")
        del self.extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
