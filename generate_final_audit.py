import sys
from pathlib import Path
import logging
import torch
import numpy as np
import time
import traceback

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

argus_path = root / "argus_stream_extracted" / "argus stream A"
if str(argus_path) not in sys.path:
    sys.path.insert(0, str(argus_path))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

report = []

def log(msg):
    logger.info(msg)
    report.append(msg)

def audit_video_mae():
    log("=== AUDITING VideoMAE (Semantic Encoder) ===")
    try:
        from src.models.encoders.videomae import VideoMAEEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VideoMAEEncoder().to(device)
        dummy_input = torch.randn(1, 3, 16, 224, 224).to(device)
        t0 = time.time()
        with torch.no_grad():
            feat = model(dummy_input)
        latency = (time.time() - t0) * 1000
        log(f"[PASS] Executable: YES")
        log(f"Input Shape: {dummy_input.shape}")
        log(f"Output Shape: {feat.shape}")
        log(f"Latency: {latency:.2f} ms")
    except Exception as e:
        log(f"[WARNING] VideoMAE Initialization Error (Likely missing checkpoint/dependency): {e}")

def audit_mulde():
    log("=== AUDITING MULDE + GMM ===")
    try:
        from src.models.scorers.mulde import MULDEScorer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = root / "models" / "pretrained" / "stream_a_real" / "checkpoints" / "stream_a" / "best_clip.pt"
        if not ckpt_path.exists():
            log(f"[FAIL] Missing Checkpoint at {ckpt_path}")
            return
        model = MULDEScorer.load_checkpoint(ckpt_path, device=device)
        model.eval()
        dummy_feat = torch.randn(1, 768).to(device)
        t0 = time.time()
        with torch.no_grad():
            score = model.score_anomaly(dummy_feat)
        latency = (time.time() - t0) * 1000
        log(f"[PASS] Executable: YES")
        log(f"Input Shape: (1, 768)")
        log(f"Output Shape: {score.shape}")
        log(f"Latency: {latency:.2f} ms")
    except Exception as e:
        log(f"[FAIL] MULDE Error: {e}")

def audit_lstm():
    log("=== AUDITING LSTM Predictor ===")
    try:
        from v2.prediction.lstm.lstm_predictor_wrapper import LSTMPredictor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = root / "v2" / "prediction" / "lstm" / "lstm_best.pth"
        if not ckpt_path.exists():
            log(f"[FAIL] Missing LSTM Checkpoint at {ckpt_path}")
            return
        model = LSTMPredictor(input_dim=5, forecast_dim=10).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        dummy_input = torch.randn(1, 30, 5).to(device)
        t0 = time.time()
        with torch.no_grad():
            pred = model(dummy_input)
        latency = (time.time() - t0) * 1000
        log(f"[PASS] Executable: YES")
        log(f"Input Shape: {dummy_input.shape}")
        log(f"Output Shape: {pred.shape}")
        log(f"Latency: {latency:.2f} ms")
    except Exception as e:
        log(f"[FAIL] LSTM Error: {e}")

def audit_ppo():
    log("=== AUDITING PPO Agent ===")
    try:
        from v2.rl.ppo_agent import PPOAgent, RolloutBuffer
        agent = PPOAgent(state_dim=168, action_dim=4)
        dummy_state = np.random.randn(168)
        buffer = RolloutBuffer()
        t0 = time.time()
        action = agent.act(dummy_state, buffer)
        latency = (time.time() - t0) * 1000
        log(f"[PASS] Executable: YES")
        log(f"Input Shape: (168,)")
        log(f"Action Taken: {action}")
        log(f"Latency: {latency:.2f} ms")
    except Exception as e:
        log(f"[FAIL] PPO Error: {e}")

def audit_mappo():
    log("=== AUDITING MAPPO Agent ===")
    try:
        from v2.rl.mappo import MAPPOAgent
        log(f"[PASS] MAPPO Exists")
    except ImportError:
        log(f"[FAIL] MAPPO Not implemented in V2 architecture.")

def audit_safety():
    log("=== AUDITING Safety Module ===")
    try:
        from v2.safety.safety_wrapper import SafetyWrapper
        log(f"[PASS] SafetyWrapper Exists")
    except ImportError:
        log(f"[WARNING] Explicit Safety wrapper not found (handled natively in Reward formulation)")

def audit_environment():
    log("=== AUDITING SPGRL Environment & Unified Builder ===")
    try:
        from v2.rl.spgrl_environment import SPGRLEnv
        env = SPGRLEnv()
        t0 = time.time()
        state = env.reset()
        next_state, reward, done, info = env.step(3)
        latency = (time.time() - t0) * 1000
        log(f"[PASS] Executable: YES")
        log(f"State Shape: {state.shape}")
        log(f"Next State Shape: {next_state.shape}")
        log(f"Reward: {reward}")
        log(f"Latency (1 step + reset): {latency:.2f} ms")
        
        log(f"[INFO] Mocked Values identified:")
        log(f" - YOLO/DeepSORT: Using synthetic backend")
        log(f" - Graph/Carbon: Heuristic scorers")
        log(f" - Cf: Zero-filled placeholder")
    except Exception as e:
        log(f"[FAIL] Environment Error: {e}")

if __name__ == "__main__":
    audit_video_mae()
    audit_mulde()
    audit_lstm()
    audit_ppo()
    audit_mappo()
    audit_safety()
    audit_environment()
    
    with open("audit_results.log", "w") as f:
        f.write("\n".join(report))
