import os
import sys
import json
import csv
import torch
import numpy as np
from pathlib import Path

# Force dummy video driver for headless env
os.environ["SDL_VIDEODRIVER"] = "dummy"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Prepare outputs
OUT_DIR = PROJECT_ROOT / "outputs" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

tensor_trace = []
callgraph = []

def log_tensor(module_name, tensor):
    if isinstance(tensor, torch.Tensor):
        shape = str(list(tensor.shape))
        dtype = str(tensor.dtype)
        device = str(tensor.device)
    elif isinstance(tensor, np.ndarray):
        shape = str(list(tensor.shape))
        dtype = str(tensor.dtype)
        device = "cpu (numpy)"
    elif isinstance(tensor, list):
        shape = f"list(len={len(tensor)})"
        dtype = "list"
        device = "cpu"
    else:
        shape = str(type(tensor))
        dtype = "scalar/other"
        device = "cpu"
        
    tensor_trace.append({
        "module_name": module_name,
        "tensor_shape": shape,
        "dtype": dtype,
        "device": device
    })

class MockMULDEScorer(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        
    def score_anomaly(self, feature):
        callgraph.append("MULDEScorer.score_anomaly")
        log_tensor("MULDEScorer_input", feature)
        # return dummy scalar tensor
        out = torch.tensor([4.5], dtype=torch.float32, device=self.device)
        log_tensor("MULDEScorer_output", out)
        return out
        
    def eval(self):
        pass

def patch_loader():
    import intelligence.perception.stream_a.loader as loader_mod
    original_load = loader_mod.load_stream_a_models
    
    def mocked_load():
        callgraph.append("loader.load_stream_a_models")
        VideoMAEFeatureExtractor, MULDEScorer = original_load()
        
        # Monkeypatch MULDEScorer
        MULDEScorer.load_checkpoint = classmethod(lambda cls, path, device="cpu": MockMULDEScorer(device=device))
        
        return VideoMAEFeatureExtractor, MULDEScorer
        
    loader_mod.load_stream_a_models = mocked_load

def run_structural_validation():
    print("Starting Phase 10: Structural Runtime Validation...")
    patch_loader()
    
    from intelligence.environments.traffic_env import TrafficEnvironment
    from intelligence.perception.stream_a.engine import ARGUSEngine
    from intelligence.perception.stream_a.provider import SyntheticRenderProvider

    callgraph.append("init_provider")
    provider = SyntheticRenderProvider()
    
    callgraph.append("init_ARGUSEngine")
    engine = ARGUSEngine(frame_provider=provider, mulde_checkpoint="mocked.pt", device="cpu")
    
    callgraph.append("init_TrafficEnvironment")
    env = TrafficEnvironment(argus_engine=engine)
    
    callgraph.append("env.reset")
    obs, info = env.reset()
    log_tensor("TrafficEnv_reset_obs", obs)
    
    print("Executing 5 simulation steps to trace callgraph...")
    for i in range(5):
        callgraph.append(f"env.step_{i}")
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        log_tensor(f"TrafficEnv_step_{i}_obs", obs)
        
    print("Writing artifacts...")
    
    # 1. runtime_callgraph.json
    with open(OUT_DIR / "runtime_callgraph.json", "w") as f:
        json.dump(callgraph, f, indent=2)
        
    # 2. tensor_shape_trace.csv
    with open(OUT_DIR / "tensor_shape_trace.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["module_name", "tensor_shape", "dtype", "device"])
        writer.writeheader()
        writer.writerows(tensor_trace)
        
    # 3. structural_validation_report.json
    report = {
        "status": "success",
        "steps_completed": 5,
        "environment": "TrafficEnvironment",
        "engine": "ARGUSEngine",
        "mocked_mulde": True,
        "observation_dim_verified": obs.shape == (28,)
    }
    with open(OUT_DIR / "structural_validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("Phase 10 complete. Artifacts written to outputs/validation/")

if __name__ == "__main__":
    run_structural_validation()
