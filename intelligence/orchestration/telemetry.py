import os
import csv
import json
import time
import logging
import psutil
from collections import deque
from pathlib import Path
from typing import Dict, Any, List

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

class StageTimer:
    """Records timings for individual components without modifying them."""
    def __init__(self, tracer, stage_name: str):
        self.tracer = tracer
        self.stage_name = stage_name
        
    def __enter__(self):
        self.start = time.time()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = (time.time() - self.start) * 1000.0
        self.tracer.record_stage_latency(self.stage_name, latency)

class TimedExtractorProxy:
    """Dynamically proxies VideoMAE and MULDE to extract individual latency metrics."""
    def __init__(self, extractor_instance, tracer):
        self._extractor = extractor_instance
        self._tracer = tracer
        
        # We proxy the internal models
        if hasattr(self._extractor, 'videomae'):
            self._extractor.videomae = self._wrap_model(self._extractor.videomae, 'videomae', 'extract_from_frames')
        if hasattr(self._extractor, 'mulde'):
            self._extractor.mulde = self._wrap_model(self._extractor.mulde, 'mulde', 'score_anomaly')
            
    def _wrap_model(self, model, stage_name: str, target_method: str):
        original_method = getattr(model, target_method)
        def wrapper(*args, **kwargs):
            with StageTimer(self._tracer, stage_name):
                return original_method(*args, **kwargs)
        setattr(model, target_method, wrapper)
        return model

class RuntimeTracer:
    """Logs the exact execution JSON trace for the paper."""
    def __init__(self, runtime_dir: Path):
        self.log_path = runtime_dir / "runtime_trace.json"
        self.history = []
        self._current_step = {}
        self.active = True
        
    def record_stage_latency(self, stage: str, latency_ms: float):
        self._current_step[f"{stage}_latency_ms"] = latency_ms
        
    def begin_step(self, step_idx: int):
        self._current_step = {
            "step": step_idx,
            "timestamp": time.time(),
        }
        
    def record_telemetry(self, key: str, value: Any):
        self._current_step[key] = value
        
    def end_step(self):
        if self._current_step:
            self.history.append(self._current_step)
            # For massive runs, you'd append-write. We'll do simple dumps for now.
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

class TensorTracer:
    """Logs tensor shapes, dtypes, and stats to CSV."""
    def __init__(self, runtime_dir: Path):
        self.log_path = runtime_dir / "tensor_trace.csv"
        self.active = True
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "tensor_name", "shape", "dtype", "device", "memory_MB", "min", "max", "mean", "std"])
            
    def log_tensor(self, step: int, name: str, tensor: Any):
        if not self.active:
            return
            
        import numpy as np
        
        try:
            if isinstance(tensor, np.ndarray):
                shape = str(tensor.shape)
                dtype = str(tensor.dtype)
                device = "cpu"
                mem = tensor.nbytes / (1024**2)
                t_min = float(np.min(tensor)) if tensor.size > 0 else 0
                t_max = float(np.max(tensor)) if tensor.size > 0 else 0
                t_mean = float(np.mean(tensor)) if tensor.size > 0 else 0
                t_std = float(np.std(tensor)) if tensor.size > 0 else 0
            elif torch is not None and isinstance(tensor, torch.Tensor):
                shape = str(list(tensor.shape))
                dtype = str(tensor.dtype)
                device = str(tensor.device)
                mem = (tensor.element_size() * tensor.nelement()) / (1024**2)
                t_min = float(tensor.min().item()) if tensor.numel() > 0 else 0
                t_max = float(tensor.max().item()) if tensor.numel() > 0 else 0
                
                # mean/std only for float types usually
                if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                    t_mean = float(tensor.mean().item()) if tensor.numel() > 0 else 0
                    t_std = float(tensor.std().item()) if tensor.numel() > 1 else 0
                else:
                    t_mean, t_std = 0.0, 0.0
            else:
                return # Skip unknown formats
                
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, name, shape, dtype, device, f"{mem:.4f}", f"{t_min:.4f}", f"{t_max:.4f}", f"{t_mean:.4f}", f"{t_std:.4f}"])
        except Exception as e:
            logger.debug(f"Failed to log tensor trace for {name}: {e}")

class BenchmarkRecorder:
    """Logs hardware utilization (CPU, RAM, GPU)."""
    def __init__(self, benchmark_dir: Path):
        self.latency_path = benchmark_dir / "latency.csv"
        self.sys_path = benchmark_dir / "system.csv"
        self.process = psutil.Process()
        
        with open(self.latency_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "total_step_ms", "env_ms", "perception_ms", "rl_ms"])
            
        with open(self.sys_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "cpu_percent", "ram_mb", "gpu_mb"])
            
    def log_latency(self, step: int, total: float, env: float, perception: float, rl: float):
        with open(self.latency_path, "a", newline="") as f:
            csv.writer(f).writerow([step, f"{total:.2f}", f"{env:.2f}", f"{perception:.2f}", f"{rl:.2f}"])
            
    def log_system(self):
        cpu = psutil.cpu_percent()
        ram = self.process.memory_info().rss / (1024**2)
        gpu = 0.0
        if torch is not None and torch.cuda.is_available():
            gpu = torch.cuda.memory_allocated() / (1024**2)
            
        with open(self.sys_path, "a", newline="") as f:
            csv.writer(f).writerow([time.time(), f"{cpu:.1f}", f"{ram:.1f}", f"{gpu:.1f}"])

class PipelineHealthMonitor:
    """Continuously verifies pipeline health."""
    def __init__(self, max_latency_ms: float = 1000.0):
        self.max_latency_ms = max_latency_ms
        self.latency_queue = deque(maxlen=100)
        
    def check_health(self, step: int, obs: Any, step_latency_ms: float):
        if step_latency_ms > self.max_latency_ms:
            logger.warning(f"Pipeline latency spike at step {step}: {step_latency_ms:.1f}ms exceeds threshold {self.max_latency_ms}ms")
            
        import numpy as np
        if isinstance(obs, np.ndarray):
            if not np.isfinite(obs).all():
                logger.error(f"FATAL: NaN or Inf detected in observation at step {step}")
                
        if torch is not None and torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024**2)
            if gpu_mem > 8000: # 8GB
                logger.warning(f"High GPU memory usage at step {step}: {gpu_mem:.1f}MB")
