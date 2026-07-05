import os
import json
from pathlib import Path
import hashlib

def get_hash(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    root = Path(__file__).resolve().parent.parent.parent
    audit_dir = root / "outputs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    # PHASE 1: REPOSITORY READINESS AUDIT
    required_files = {
        "train.py": root / "train.py",
        "SumoEnvironment": root / "intelligence" / "environments" / "sumo_env.py",
        "ARGUSEngine": root / "intelligence" / "perception" / "stream_a" / "engine.py",
        "FrameProvider": root / "intelligence" / "perception" / "stream_a" / "provider.py",
        "FrameBuffer": root / "intelligence" / "perception" / "stream_a" / "provider.py",
        "StreamAOnlineExtractor": root / "intelligence" / "perception" / "stream_a" / "extractor.py",
        "HybridStateBuilder": root / "intelligence" / "perception" / "stream_a" / "state.py",
        "RLObservationMapper": root / "intelligence" / "perception" / "stream_a" / "mapper.py",
        "AssetManager": root / "intelligence" / "orchestration" / "asset_manager.py",
        "ExperimentRecorder": root / "intelligence" / "orchestration" / "experiment.py",
        "RuntimeTracer": root / "intelligence" / "orchestration" / "telemetry.py",
        "TensorTracer": root / "intelligence" / "orchestration" / "telemetry.py",
        "BenchmarkRecorder": root / "intelligence" / "orchestration" / "telemetry.py",
        "PipelineHealthMonitor": root / "intelligence" / "orchestration" / "telemetry.py",
        "EnvironmentValidator": root / "intelligence" / "environments" / "validation.py"
    }
    
    readiness = {}
    for name, path in required_files.items():
        readiness[name] = {
            "exists": path.exists(),
            "path": str(path.relative_to(root)) if path.exists() else None
        }
    
    with open(audit_dir / "repository_readiness.json", "w") as f:
        json.dump(readiness, f, indent=4)
        
    # PHASE 2: ASSET DISCOVERY
    assets = {
        "VideoMAE_checkpoint": root / "argus_stream_extracted" / "argus stream A" / "checkpoints" / "best.pt",
        "MULDE_checkpoint": root / "argus_stream_extracted" / "argus stream A" / "checkpoints" / "best.pt",
        "GMM_model": root / "argus_stream_extracted" / "argus stream A" / "checkpoints" / "best.pkl",
        "UA-DETRAC_video": root / "datasets" / "ua_detrac" / "MVI_20011.mp4",
        "config_file": root / "config" / "default.yaml"
    }
    
    asset_inventory = {}
    for name, path in assets.items():
        exists = path.exists()
        asset_inventory[name] = {
            "exists": exists,
            "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            "size_bytes": os.path.getsize(path) if exists else None,
            "sha256": get_hash(path) if exists else None
        }
        
    with open(root / "asset_inventory.json", "w") as f:
        json.dump(asset_inventory, f, indent=4)
        
    validation_report = {
        "status": "FAILED" if not all(v["exists"] for v in asset_inventory.values()) else "PASSED",
        "missing_assets": [k for k, v in asset_inventory.items() if not v["exists"]]
    }
    
    with open(root / "asset_validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=4)

if __name__ == "__main__":
    main()
