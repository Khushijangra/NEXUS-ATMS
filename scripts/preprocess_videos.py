import os
import cv2
import glob
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = Path("data/raw/ua_detrac")
OUTPUT_DIR = Path("data/processed/ua_detrac")
TARGET_SIZE = (224, 224)

def preprocess_videos():
    if not INPUT_DIR.exists():
        logging.error(f"Input directory missing. Blocking execution: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_files = list(INPUT_DIR.rglob("*.mp4")) + list(INPUT_DIR.rglob("*.avi"))
    
    if not video_files:
        logging.error(f"No videos found in {INPUT_DIR}. Please place UA-DETRAC videos here.")
        return

    logging.info(f"Found {len(video_files)} videos. Starting preprocessing...")
    
    for idx, video_path in enumerate(video_files):
        out_path = OUTPUT_DIR / f"{video_path.stem}_processed.mp4"
        
        # Resumable execution
        if out_path.exists() and out_path.stat().st_size > 1024:
            logging.info(f"[{idx+1}/{len(video_files)}] Skipping already processed: {out_path.name}")
            continue
            
        logging.info(f"[{idx+1}/{len(video_files)}] Processing {video_path.name}...")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Failed to open {video_path}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(out_path), fourcc, fps, TARGET_SIZE)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize for VideoMAE
            frame_resized = cv2.resize(frame, TARGET_SIZE)
            out.write(frame_resized)
            
        cap.release()
        out.release()
        logging.info(f"Saved {out_path.name}")

if __name__ == "__main__":
    preprocess_videos()
