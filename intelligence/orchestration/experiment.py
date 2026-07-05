import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExperimentRecorder:
    """
    Manages the lifecycle and standardized directory structure for experiments.
    Ensures reproducibility by organizing manifests, configs, traces, and models.
    """
    
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = Path(base_log_dir)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.experiment_dir = self.base_log_dir / f"experiment_{timestamp}"
        
        # Subdirectories
        self.runtime_dir = self.experiment_dir / "runtime"
        self.benchmark_dir = self.experiment_dir / "benchmark"
        self.validation_dir = self.experiment_dir / "validation"
        self.models_dir = self.experiment_dir / "models"
        self.tensorboard_dir = self.experiment_dir / "tensorboard"
        
        self._create_structure()
        
    def _create_structure(self):
        """Creates the standardized folder structure."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(exist_ok=True)
        self.benchmark_dir.mkdir(exist_ok=True)
        self.validation_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.tensorboard_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized experiment directory: {self.experiment_dir}")
        
    def record_config(self, config: Dict[str, Any], raw_yaml_path: Optional[str] = None):
        """Saves the configuration state."""
        # Dump structured JSON representation
        with open(self.experiment_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        # If original yaml is provided, copy it directly
        if raw_yaml_path and os.path.exists(raw_yaml_path):
            shutil.copy(raw_yaml_path, self.experiment_dir / "config.yaml")
            
    def record_git_commit(self, project_root: Path):
        """Attempts to fetch and record the current git commit."""
        import subprocess
        commit_file = self.experiment_dir / "git_commit.txt"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                cwd=project_root, capture_output=True, text=True, check=True
            )
            commit_hash = result.stdout.strip()
            
            diff_result = subprocess.run(
                ["git", "diff", "--stat"], 
                cwd=project_root, capture_output=True, text=True, check=True
            )
            diff_stat = diff_result.stdout.strip()
            
            with open(commit_file, "w", encoding="utf-8") as f:
                f.write(f"Commit: {commit_hash}\n")
                f.write(f"Uncommitted Changes:\n{diff_stat}\n")
        except Exception as e:
            with open(commit_file, "w", encoding="utf-8") as f:
                f.write(f"Git commit could not be recorded: {e}\n")
                
    def get_dir(self, name: str) -> Path:
        """Returns the path to a specific subdirectory."""
        dirs = {
            "root": self.experiment_dir,
            "runtime": self.runtime_dir,
            "benchmark": self.benchmark_dir,
            "validation": self.validation_dir,
            "models": self.models_dir,
            "tensorboard": self.tensorboard_dir
        }
        return dirs[name]
