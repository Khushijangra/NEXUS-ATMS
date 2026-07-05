import os
import json
import csv
from pathlib import Path
from datetime import datetime

def safe_load_torch(filepath):
    try:
        import torch
        # Load safely to CPU without running any code
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        metadata = {
            "type": str(type(data)),
            "keys": list(data.keys()) if isinstance(data, dict) else [],
            "has_state_dict": 'state_dict' in data if isinstance(data, dict) else False,
            "has_optimizer": 'optimizer_states' in data if isinstance(data, dict) else False,
            "epoch": data.get('epoch') if isinstance(data, dict) else None
        }
        return metadata
    except Exception as e:
        return {"error": str(e)}

def safe_load_pickle(filepath):
    try:
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        metadata = {
            "type": str(type(data)),
            "is_gmm": "GaussianMixture" in str(type(data)),
            "keys": list(data.keys()) if isinstance(data, dict) else []
        }
        return metadata
    except Exception as e:
        return {"error": str(e)}

def main():
    root = Path(__file__).resolve().parent.parent.parent
    audit_dir = root / "outputs" / "forensic"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    exclude_dirs = {'.git', '__pycache__', 'venv', '.env', 'node_modules', '.pytest_cache'}
    
    inventory = {"total_files": 0, "total_size_bytes": 0, "directories": [], "extensions": {}}
    models = []
    lightning_checkpoints = []
    videos = []
    configs = []
    outputs = []
    gmm_models = []
    torch_metadata = {}
    
    model_exts = {'.pt', '.pth', '.ckpt', '.bin', '.pkl', '.joblib', '.onnx', '.safetensors'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.jpg', '.png'}
    config_exts = {'.yaml', '.yml', '.json', '.toml', '.ini'}
    output_dirs = {'lightning_logs', 'wandb', 'tensorboard', 'mlruns', 'outputs', 'logs'}
    
    print("Starting deep recursive forensic sweep...")
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        current_dir = Path(dirpath)
        rel_dir = current_dir.relative_to(root)
        
        inventory["directories"].append(str(rel_dir))
        
        for f in filenames:
            filepath = current_dir / f
            try:
                stat = filepath.stat()
                size = stat.st_size
                ext = filepath.suffix.lower()
                rel_path = str(filepath.relative_to(root))
                
                inventory["total_files"] += 1
                inventory["total_size_bytes"] += size
                inventory["extensions"][ext] = inventory["extensions"].get(ext, 0) + 1
                
                file_info = {
                    "path": rel_path,
                    "size": size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "extension": ext
                }
                
                if ext in model_exts:
                    models.append(file_info)
                    if ext in {'.pt', '.pth', '.ckpt'}:
                        if 'lightning_logs' in current_dir.parts or 'checkpoints' in current_dir.parts:
                            lightning_checkpoints.append(file_info)
                        torch_metadata[rel_path] = safe_load_torch(filepath)
                    elif ext == '.pkl':
                        gmm_info = safe_load_pickle(filepath)
                        gmm_models.append({"path": rel_path, "metadata": gmm_info})
                
                if ext in video_exts:
                    videos.append(file_info)
                    
                if ext in config_exts:
                    configs.append(file_info)
                    
                if any(out_dir in current_dir.parts for out_dir in output_dirs):
                    outputs.append(file_info)
                    
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    # Output generation
    print("Writing forensic reports...")
    with open(audit_dir / "workspace_inventory.json", "w") as f:
        json.dump(inventory, f, indent=4)
        
    with open(audit_dir / "model_artifacts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size", "modified", "extension"])
        writer.writeheader()
        writer.writerows(models)
        
    with open(audit_dir / "lightning_checkpoint_report.json", "w") as f:
        json.dump(lightning_checkpoints, f, indent=4)
        
    with open(audit_dir / "checkpoint_metadata.json", "w") as f:
        json.dump(torch_metadata, f, indent=4)
        
    with open(audit_dir / "gmm_inventory.json", "w") as f:
        json.dump(gmm_models, f, indent=4)
        
    with open(audit_dir / "video_inventory.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size", "modified", "extension"])
        writer.writeheader()
        writer.writerows(videos)
        
    with open(audit_dir / "experiment_outputs.json", "w") as f:
        json.dump(outputs, f, indent=4)
        
    with open(audit_dir / "configuration_inventory.json", "w") as f:
        json.dump(configs, f, indent=4)
        
    print("Audit complete.")

if __name__ == "__main__":
    main()
