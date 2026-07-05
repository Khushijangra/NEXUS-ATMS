import os
import json
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent.parent
    audit_dir = root / "outputs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    train_script = root / "train.py"
    
    cmd = [
        "python", str(train_script),
        "--demo",
        "--perception-mode", "replay",
        "--mulde-checkpoint", "argus_stream_extracted/argus stream A/checkpoints/best.pt",
        "--video-path", "datasets/ua_detrac/MVI_20011.mp4"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    
    if result.returncode == 0:
        outcome = "Execution succeeds"
    elif result.returncode == 1 and "Asset validation failed" in result.stdout:
        outcome = "Execution halts gracefully with AssetManager"
    else:
        outcome = "Unexpected failure"
        
    report = {
        "expected_outcomes": ["Execution succeeds", "Execution halts gracefully with AssetManager"],
        "actual_outcome": outcome,
        "exit_code": result.returncode,
        "exception": result.stderr if result.returncode != 0 else None,
        "log_trace": result.stdout
    }
    
    with open(audit_dir / "production_execution_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Outcome: {outcome}")

if __name__ == "__main__":
    main()
