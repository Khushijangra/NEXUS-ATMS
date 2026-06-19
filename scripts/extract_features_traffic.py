import os
import sys
import torch
import numpy as np
import logging
import cv2
from pathlib import Path

# Add ARGUS directory to path for imports
sys.path.append(os.path.abspath(r"argus_stream_extracted\argus stream A"))

try:
    from transformers import VideoMAEFeatureExtractor, VideoMAEModel
except ImportError:
    logging.error("Missing transformers module. Run: pip install transformers")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = Path("data/processed/ua_detrac")
OUTPUT_DIR = Path("data/features/ua_detrac/videomae")
BATCH_SIZE = 1 # Strictly 1 for RTX 2050 4GB

def extract_features():
    if not INPUT_DIR.exists():
        logging.error(f"Input directory missing. Run preprocess_videos.py first: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_files = list(INPUT_DIR.rglob("*.mp4"))
    
    if not video_files:
        logging.error("No processed videos found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load Model (VideoMAEv2-Base)
    model_id = "MCG-NJU/videomae-base"
    logging.info(f"Loading {model_id}...")
    processor = VideoMAEFeatureExtractor.from_pretrained(model_id)
    # Use FP16 for RTX 2050 memory savings
    model = VideoMAEModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    
    for idx, video_path in enumerate(video_files):
        out_path = OUTPUT_DIR / f"{video_path.stem}.npy"
        
        # Resumable execution
        if out_path.exists():
            logging.info(f"[{idx+1}/{len(video_files)}] Skipping extracted: {out_path.name}")
            continue
            
        logging.info(f"[{idx+1}/{len(video_files)}] Extracting {video_path.name}...")
        
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        features = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            # Process in 16-frame chunks
            if len(frames) == 16:
                inputs = processor(list(frames), return_tensors="pt")
                inputs = {k: v.to(device).half() for k, v in inputs.items()} # Cast to FP16
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Average pooling over spatial and temporal tokens [1, 768]
                    chunk_feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                features.append(chunk_feat)
                frames = [] # clear buffer
                
        cap.release()
        
        if features:
            final_features = np.vstack(features)
            np.save(str(out_path), final_features)
            logging.info(f"Saved {out_path.name} with shape {final_features.shape}")
        else:
            logging.warning(f"Video {video_path.name} had too few frames.")

if __name__ == "__main__":
    extract_features()
