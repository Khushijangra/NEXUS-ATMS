import sys
import json
import time
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock the deep layers so we can trace without actual weights
def mock_load_stream_a():
    class MockVideoMAE:
        def __init__(self, device):
            self.device = device
            
        def extract_from_frames(self, frames, batch_size=1):
            # Log the shape to trace
            trace["VideoMAE_input_shape"] = list(np.array(frames).shape)
            out = np.random.randn(1, 768).astype(np.float32)
            trace["VideoMAE_output_shape"] = list(out.shape)
            return out

    class MockMULDE:
        def __init__(self, device):
            self.device = device
            
        @classmethod
        def load_checkpoint(cls, path, device="cpu"):
            return cls(device=device)
            
        def eval(self):
            pass
            
        def score_anomaly(self, feature):
            # Log the scalar
            score = 0.842
            trace["MULDE_anomaly_score"] = score
            import torch
            return torch.tensor([score])
            
    return MockVideoMAE, MockMULDE

trace = {}

def run_verification():
    import train
    
    # We will run 1 single step.
    # To do this cleanly, we intercept the SB3 learn method.
    original_learn = None
    
    def mock_learn(self, *args, **kwargs):
        # We step the environment manually exactly 1 time to trace it
        start_time = time.time()
        obs, info = self.env.reset()
        action = [self.env.action_space.sample()]
        obs, reward, done, info = self.env.step(action)
        
        trace["env_step_latency_sec"] = time.time() - start_time
        trace["final_observation_shape"] = list(obs.shape)
        trace["final_observation_vector"] = [float(x) for x in obs[0]]
        
        # Stop training by doing nothing
        return
        
    with patch("intelligence.perception.stream_a.loader.load_stream_a_models", side_effect=mock_load_stream_a):
        with patch("stable_baselines3.PPO.learn", new=mock_learn):
            with patch("train.load_config") as mock_load_config:
                import yaml
                with open(PROJECT_ROOT / "configs" / "default.yaml") as f:
                    cfg = yaml.safe_load(f)
                cfg["environment"]["network_file"] = "simulation/networks/single_intersection.net.xml"
                cfg["environment"]["route_file"] = "simulation/networks/scenarios/rush_hour.rou.xml"
                mock_load_config.return_value = cfg
                
                sys.argv = [
                    "train.py",
                    "--demo",
                    "--perception-mode", "synthetic",
                    "--mulde-checkpoint", "mocked.pt"
                ]
                
                try:
                    train.main()
                except SystemExit:
                    pass
                
    # Dump trace
    out_dir = PROJECT_ROOT / "outputs" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runtime_trace.json", "w") as f:
        json.dump(trace, f, indent=2)
        
    print(json.dumps(trace, indent=2))

if __name__ == "__main__":
    run_verification()
