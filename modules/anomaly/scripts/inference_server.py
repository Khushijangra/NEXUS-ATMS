import argparse
import json
import logging
import time
import zmq
import numpy as np
import torch
import os
import sys

# Ensure ARGUS imports work when run from ARGUS root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.models.scorers.mulde import MULDEScorer
    from src.models.backbones.videomae import VideoMAEFeatureExtractor
    MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import ARGUS models: {e}. Will run in dummy mode.")
    MODELS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def serve(port=5555, checkpoint_path=None, dummy_mode=False):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    
    logging.info(f"Inference Server started on tcp://*:{port}")
    
    scorer = None
    extractor = None
    
    if not dummy_mode and MODELS_AVAILABLE:
        logging.info("Loading models for Validation/Stress Testing mode...")
        if checkpoint_path and os.path.exists(checkpoint_path):
            scorer = MULDEScorer(feature_dim=768, hidden_dim=4096, gmm_components=5)
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in state_dict:
                scorer.load_state_dict(state_dict['model_state_dict'])
            else:
                scorer.load_state_dict(state_dict)
            scorer.eval()
            if torch.cuda.is_available():
                scorer = scorer.cuda()
            logging.info("MULDEScorer loaded successfully.")
            
            # Note: During RL training, the VideoMAE isn't run frame-by-frame to save VRAM.
            # This full pipeline is reserved for stress testing / real validation.
            # extractor = VideoMAEFeatureExtractor()
            # logging.info("VideoMAEFeatureExtractor loaded successfully.")
        else:
            logging.warning("No checkpoint provided or path invalid. Running in surrogate/synthetic mode.")
    else:
        logging.info("Running in Surrogate/Synthetic mode for RL Training.")

    while True:
        try:
            message = socket.recv_json()
            t0 = time.time()
            
            action = message.get("action", "get_score")
            req_context = message.get("context", "synthetic")
            incident_type = message.get("incident_type", "none")
            
            anomaly_score = 0.0
            anomaly_flag = 0
            
            if action == "get_score":
                if req_context == "synthetic" or scorer is None:
                    # Synthetic Surrogate Mode for RL Training
                    if incident_type in ["stopped_vehicle", "lane_blockage", "intersection_obstruction"]:
                        anomaly_score = float(np.clip(np.random.normal(0.85, 0.05), 0.0, 1.0))
                        anomaly_flag = 1
                    else:
                        anomaly_score = float(np.clip(np.random.normal(0.1, 0.05), 0.0, 1.0))
                        anomaly_flag = 0
                else:
                    # Validation / Stress Testing Mode
                    # Here we would ingest real pre-extracted features or run the extractor
                    # For now, placeholder for the full validation pipeline path.
                    pass
            
            processing_time_ms = (time.time() - t0) * 1000.0
            
            response = {
                "status": "success",
                "anomaly_score": anomaly_score,
                "anomaly_flag": anomaly_flag,
                "incident_type": incident_type,
                "processing_time_ms": processing_time_ms
            }
            
            socket.send_json(response)
            
        except KeyboardInterrupt:
            logging.info("Shutting down inference server...")
            break
        except Exception as e:
            logging.error(f"Error processing request: {e}")
            socket.send_json({"status": "error", "message": str(e)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dummy", action="store_true", help="Run without loading models (Synthetic only)")
    args = parser.parse_args()
    
    serve(args.port, args.checkpoint, args.dummy)
