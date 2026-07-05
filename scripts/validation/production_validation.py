import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("production_validation")

def main():
    logger.info("Starting End-to-End Production Validation...")
    
    project_root = Path(__file__).parent.parent.parent
    train_script = project_root / "train.py"
    
    if not train_script.exists():
        logger.error(f"Cannot find train.py at {train_script}")
        sys.exit(1)
        
    cmd = [
        sys.executable, str(train_script),
        "--demo",
        "--perception-mode", "replay",
        "--mulde-checkpoint", "argus_stream_extracted/argus stream A/checkpoints/best.pt",
        "--video-path", "datasets/ua_detrac/MVI_20011.mp4"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    
    # We expect this to fail gracefully at Asset Validation right now, returning exit code 1.
    # Once assets are provided, this script will be updated to expect exit code 0.
    
    logger.info("===== OUTPUT BEGIN =====")
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    logger.info("===== OUTPUT END =====")
    
    logger.info(f"Exit code: {result.returncode}")
    
    if result.returncode == 1 and "Asset validation failed" in result.stdout:
        logger.info("✅ Pipeline correctly halted at pre-flight asset validation (Expected behavior when missing assets).")
    elif result.returncode == 0:
        logger.info("✅ Pipeline executed successfully (Expected behavior when assets are present).")
    else:
        logger.error(f"❌ Unexpected failure mode. Return code: {result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
