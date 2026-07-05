"""
Asset Management Subsystem
Handles discovery, validation, and structured error reporting for production assets
like model checkpoints and video datasets.
"""

import os
import json
import logging
import platform
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("asset_manager")

class CheckpointManager:
    """Validates PyTorch checkpoints before loading."""
    
    @staticmethod
    def validate_checkpoint(path: Path) -> Tuple[bool, str]:
        if not path.exists():
            return False, f"Missing file: {path}"
        if not path.is_file():
            return False, f"Not a file: {path}"
        if path.stat().st_size < 1024:
            return False, f"File too small to be a valid checkpoint: {path}"
            
        # Basic extension checks
        if path.suffix not in ['.pt', '.pth', '.pkl']:
            return False, f"Invalid checkpoint extension '{path.suffix}' for: {path}"
            
        # Deep validation if possible (only runs if torch is available and file is meant to be a PyTorch checkpoint)
        if torch is not None and path.suffix in ['.pt', '.pth']:
            try:
                # Load strictly onto CPU to avoid CUDA initialization overhead just for validation
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                
                # Verify it's a state dictionary or object containing standard keys
                if not isinstance(ckpt, dict):
                    return False, f"Invalid checkpoint format: expected dict, got {type(ckpt).__name__}"
                
                # Check for required structural keys (specific to MULDE expected structure)
                # Note: If the user provides a custom checkpoint, they might need 'state_dict'
                # or 'model_state_dict'. We do a soft check here.
                if 'state_dict' not in ckpt and not any(k.startswith('module.') for k in ckpt.keys()) and not any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                    return False, f"Checkpoint does not appear to contain valid tensor weights."
                    
                # We could add explicit feature dimension checking if the metadata exists in the checkpoint
                if 'config' in ckpt and isinstance(ckpt['config'], dict):
                    cfg = ckpt['config']
                    if 'feature_dim' in cfg and cfg['feature_dim'] != 768:
                        return False, f"Incompatible feature dimension: {cfg['feature_dim']} (expected 768)"
            except Exception as e:
                return False, f"Failed to parse checkpoint {path}: {str(e)}"
                
        # Optional: SHA256 checksum (can be slow for large files, so we could skip or just do metadata hash)
        
        return True, "Valid"

class AssetManager:
    """Production Asset Manager for experiments."""
    
    def __init__(self, config_paths: Dict[str, str], project_root: Optional[Path] = None):
        self.project_root = project_root or Path(os.getcwd())
        self.config_paths = config_paths
        
        # Detected and validated paths
        self.resolved_assets: Dict[str, Path] = {}
        self.validation_errors: Dict[str, str] = {}
        
    def discover_and_validate(self) -> bool:
        """Discovers files and populates validation structures."""
        self.resolved_assets.clear()
        self.validation_errors.clear()
        
        # 1. Video Dataset (if configured)
        video_path_str = self.config_paths.get("video_dataset")
        if video_path_str and video_path_str.lower() != "none":
            video_path = self.project_root / video_path_str
            if not video_path.exists():
                self.validation_errors["video_dataset"] = f"Video not found: {video_path}"
            elif video_path.stat().st_size == 0:
                self.validation_errors["video_dataset"] = f"Video file is empty: {video_path}"
            else:
                self.resolved_assets["video_dataset"] = video_path
                
        # 2. MULDE Checkpoint
        mulde_path_str = self.config_paths.get("mulde_checkpoint")
        if mulde_path_str and mulde_path_str.lower() != "none":
            mulde_path = self.project_root / mulde_path_str
            is_valid, msg = CheckpointManager.validate_checkpoint(mulde_path)
            if not is_valid:
                self.validation_errors["mulde_checkpoint"] = msg
            else:
                self.resolved_assets["mulde_checkpoint"] = mulde_path
                
        return len(self.validation_errors) == 0
        
    def get_asset(self, key: str) -> Optional[str]:
        """Retrieve a validated asset path as string."""
        path = self.resolved_assets.get(key)
        return str(path) if path else None

    def generate_validation_report(self, output_dir: Path) -> Path:
        """Generates a pre-flight asset validation report."""
        report = {
            "status": "PASS" if not self.validation_errors else "FAIL",
            "resolved_assets": {k: str(v) for k, v in self.resolved_assets.items()},
            "missing_or_invalid": self.validation_errors,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "asset_validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report_path
        
    def generate_experiment_manifest(self, output_dir: Path, cli_args: Dict[str, Any], raw_config: Dict[str, Any]) -> Path:
        """Generates the experiment_manifest.json containing the hardware/software snapshot."""
        try:
            import traci
            sumo_version = traci.getVersion()
        except Exception:
            sumo_version = "Unknown"
            
        manifest = {
            "environment": {
                "os": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__ if torch else "Not Installed",
                "cuda_available": torch.cuda.is_available() if torch else False,
                "sumo_version": sumo_version
            },
            "assets": {k: str(v) for k, v in self.resolved_assets.items()},
            "cli_arguments": cli_args,
            "configuration": raw_config
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "experiment_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path
