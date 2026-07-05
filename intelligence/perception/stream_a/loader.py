import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Absolute path to the standalone Stream A repository
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
STREAM_A_PATH = PROJECT_ROOT / "argus_stream_extracted" / "argus stream A"

_is_loaded = False

def load_stream_a_models():
    """
    Injects the standalone repository into sys.path and returns the required model classes.
    Ensures that the path is only injected when explicitly needed.
    """
    global _is_loaded
    
    stream_a_str = str(STREAM_A_PATH)
    
    if stream_a_str not in sys.path:
        sys.path.insert(0, stream_a_str)
        _is_loaded = True
        logger.info(f"Injected Stream A path into sys.path: {stream_a_str}")
        
    try:
        from src.models.backbones.videomae import VideoMAEFeatureExtractor
        from src.models.scorers.mulde import MULDEScorer
        return VideoMAEFeatureExtractor, MULDEScorer
    except ImportError as e:
        logger.error(f"Failed to load Stream A models from {stream_a_str}. Error: {e}")
        raise
