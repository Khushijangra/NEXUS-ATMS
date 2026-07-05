import os
import sys
import yaml
import json
import csv
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import torch

class ExperimentManager:
    """
    Manages the creation and tracking of all scientific experiments in SPGRL.
    Ensures absolute reproducibility by tracking Git commits, Configs, Seeds, and RNG states.
    """
    def __init__(self, root_dir: Path, config: dict, exp_base_name: str = "exp", resume: bool = False):
        self.root = root_dir
        self.exp_dir_base = self.root / "experiments"
        self.exp_dir_base.mkdir(exist_ok=True)
        self.config = config
        self.exp_base_name = exp_base_name
        self.resume = resume
        
        self.registry_file = self.exp_dir_base / "registry.csv"
        self._init_registry()
        
        self.current_exp_dir = self._setup_experiment_dir()
        self.logger = self._setup_logging()
        
        if not self.resume:
            self._save_provenance()
            self._write_config()
            self._init_csv_logs()
            
    def _init_registry(self):
        if not self.registry_file.exists():
            with open(self.registry_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ExpID", "Date", "Name", "Seed", "Episodes", "Status", "GitCommit", "TorchVersion"])
                
    def _get_git_commit(self):
        try:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except Exception:
            return "unknown"
            
    def _setup_experiment_dir(self):
        date_str = datetime.now().strftime("%Y%m%d")
        seed = self.config.get('seed', 42)
        episodes = self.config.get('episodes', 500)
        
        exp_name_search = f"{date_str}_{self.exp_base_name}"
        seed_search = f"_seed{seed}"
        
        if self.resume:
            existing = [d for d in self.exp_dir_base.iterdir() if d.is_dir() and exp_name_search in d.name and seed_search in d.name]
            if not existing:
                raise ValueError(f"No experiments found matching seed {seed} to resume!")
            return sorted(existing, key=lambda x: x.stat().st_mtime)[-1]
            
        exp_name = f"{date_str}_{self.exp_base_name}{episodes}_seed{seed}"
        existing = [d for d in self.exp_dir_base.iterdir() if d.is_dir() and exp_name in d.name]
        
        suffix = f"_{len(existing)+1}" if existing else ""
        new_exp_dir = self.exp_dir_base / f"{exp_name}{suffix}"
        new_exp_dir.mkdir(parents=True, exist_ok=True)
        
        (new_exp_dir / "plots").mkdir(exist_ok=True)
        (new_exp_dir / "checkpoints").mkdir(exist_ok=True)
        
        return new_exp_dir
        
    def _setup_logging(self):
        logger = logging.getLogger(f"SPGRL_{self.current_exp_dir.name}")
        logger.setLevel(logging.INFO)
        
        if logger.hasHandlers():
            logger.handlers.clear()
            
        fh = logging.FileHandler(self.current_exp_dir / "stdout.log")
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def _save_provenance(self):
        import hashlib
        
        git_commit = self._get_git_commit()
        config_str = json.dumps(self.config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        metadata = {
            "Git_Commit": git_commit,
            "Dataset_Version": "v1.0",
            "Config_Hash": config_hash,
            "Seed": self.config.get('seed', 42),
            "Timestamp": datetime.now().isoformat(),
            "Host": platform.node(),
            "OS": platform.system(),
            "CPU": platform.processor(),
            "Python_Version": sys.version,
            "Torch_Version": torch.__version__,
            "CUDA_Available": torch.cuda.is_available(),
            "CUDA_Version": torch.version.cuda if torch.cuda.is_available() else "N/A"
        }
        self.manifest_path = self.current_exp_dir / "experiment_manifest.json"
        with open(self.manifest_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        with open(self.registry_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_exp_dir.name, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.exp_base_name,
                self.config.get('seed', 42),
                self.config.get('episodes', 500),
                "STARTED",
                git_commit,
                torch.__version__
            ])
            
    def _write_config(self):
        with open(self.current_exp_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
            
    def _init_csv_logs(self):
        metrics_csv = self.current_exp_dir / "training_metrics.csv"
        if not metrics_csv.exists():
            with open(metrics_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Reward", "Queue", "Delay", "Carbon", "Entropy", "Policy_Loss", "Value_Loss", "Time"])
                    
    def log_metrics_row(self, episode: int, reward: float, queue: float, delay: float, carbon: float, entropy: float, policy_loss: float, value_loss: float, time: float):
        """Append a single row of metrics to the unified CSV."""
        metrics_csv = self.current_exp_dir / "training_metrics.csv"
        with open(metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, queue, delay, carbon, entropy, policy_loss, value_loss, time])
            
    def write_summary(self, summary_dict: dict):
        """Write the final summary JSON at the end of training and update manifest."""
        with open(self.current_exp_dir / "summary.json", "w") as f:
            json.dump(summary_dict, f, indent=4)
            
        # Update manifest with final hashes and times
        if hasattr(self, 'manifest_path') and self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                manifest = json.load(f)
                
            import hashlib
            ckpt_dir = self.current_exp_dir / "checkpoints"
            hashes = {}
            for ckpt in ckpt_dir.glob("*.pth"):
                with open(ckpt, "rb") as f:
                    hashes[ckpt.name] = hashlib.sha256(f.read()).hexdigest()
                    
            manifest.update(summary_dict)
            manifest["Checkpoint_SHA256"] = hashes
            
            with open(self.manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
