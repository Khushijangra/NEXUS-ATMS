"""
Frame Provider abstractions and rolling buffer.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import cv2
import os
import glob
from collections import deque

class FrameProvider(ABC):
    """Abstract interface for video frame acquisition, decoupling perception from source."""
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Returns the next RGB frame as a numpy array, or None if unavailable."""
        pass


class ReplayFrameProvider(FrameProvider):
    """Yields frames from a directory of pre-extracted images or an MP4 file."""
    
    def __init__(self, source_path: str, fps: float = 30.0):
        self.source_path = source_path
        self.fps = fps
        self._cap = None
        self._image_paths = []
        self._idx = 0
        
        if os.path.isdir(source_path):
            self._is_video = False
            # load sorted jpg/pngs
            self._image_paths = sorted(
                glob.glob(os.path.join(source_path, "*.jpg")) + 
                glob.glob(os.path.join(source_path, "*.png"))
            )
            if not self._image_paths:
                raise ValueError(f"No images found in {source_path}")
        elif os.path.isfile(source_path):
            self._is_video = True
            self._cap = cv2.VideoCapture(source_path)
            if not self._cap.isOpened():
                raise ValueError(f"Could not open video file {source_path}")
        else:
            raise ValueError(f"Invalid source_path: {source_path}")

    def get_frame(self) -> Optional[np.ndarray]:
        if self._is_video:
            if self._cap is None:
                return None
            ret, frame = self._cap.read()
            if not ret:
                # Loop video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            if self._idx >= len(self._image_paths):
                self._idx = 0  # loop
            path = self._image_paths[self._idx]
            self._idx += 1
            frame = cv2.imread(path)
            if frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class SyntheticRenderProvider(FrameProvider):
    """A fallback provider that generates synthetic frames (e.g. for CI testing without data)."""
    def get_frame(self) -> Optional[np.ndarray]:
        # Generate a dummy gray frame 224x224
        frame = np.full((224, 224, 3), 128, dtype=np.uint8)
        return frame


class FrameBuffer:
    """Maintains a rolling window of frames for VideoMAE clip extraction."""
    
    def __init__(self, clip_length: int = 16, stride: int = 4):
        self.clip_length = clip_length
        self.stride = stride
        self.buffer_size = (clip_length - 1) * stride + 1
        self._buffer = deque(maxlen=self.buffer_size)
    
    def push(self, frame: np.ndarray) -> None:
        """Pushes a new RGB frame into the rolling buffer."""
        # Ensure frame is 224x224 as expected by VideoMAE
        if frame.shape[:2] != (224, 224):
            frame = cv2.resize(frame, (224, 224))
        self._buffer.append(frame)
        
    def is_ready(self) -> bool:
        """Checks if enough frames are buffered to extract a clip."""
        return len(self._buffer) == self.buffer_size
        
    def get_clip(self) -> np.ndarray:
        """Extracts the strided 16-frame clip as a Numpy array [16, 224, 224, 3]."""
        if not self.is_ready():
            raise RuntimeError("Buffer not full")
        # Extract every `stride`-th frame
        frames = [self._buffer[i] for i in range(0, self.buffer_size, self.stride)]
        return np.stack(frames, axis=0)
