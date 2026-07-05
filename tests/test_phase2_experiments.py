import pytest
import os
import sys
import yaml
import json
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.experiments.experiment_manager import ExperimentManager

@pytest.fixture(scope="module")
def mock_config():
    return {"seed": 999, "episodes": 2}
    
@pytest.fixture(scope="module")
def exp_manager(mock_config):
    return ExperimentManager(root, mock_config, exp_base_name="pytest_mock")

def test_experiment_dir_created(exp_manager):
    assert exp_manager.current_exp_dir.exists()
    assert (exp_manager.current_exp_dir / "plots").exists()
    assert (exp_manager.current_exp_dir / "checkpoints").exists()

def test_metadata_generated(exp_manager):
    meta_path = exp_manager.current_exp_dir / "metadata.json"
    assert meta_path.exists()
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert "Git_Commit" in meta
    assert "Seed" in meta
    assert meta["Seed"] == 999
    
def test_config_saved(exp_manager):
    config_path = exp_manager.current_exp_dir / "config.yaml"
    assert config_path.exists()
    with open(config_path, "r") as f:
        conf = yaml.safe_load(f)
    assert conf["episodes"] == 2
    
def test_registry_updated(exp_manager):
    registry_path = exp_manager.registry_file
    assert registry_path.exists()
    
    found = False
    with open(registry_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ExpID"] == exp_manager.current_exp_dir.name:
                found = True
                assert row["Seed"] == "999"
                break
    assert found
    
def test_csv_initialization(exp_manager):
    csvs = ["reward.csv", "queue.csv", "delay.csv", "carbon.csv", "entropy.csv", "loss.csv", "safety.csv"]
    for c in csvs:
        p = exp_manager.current_exp_dir / c
        assert p.exists()
        
def test_metric_logging(exp_manager):
    exp_manager.log_metric("reward", 1, 42.0)
    p = exp_manager.current_exp_dir / "reward.csv"
    
    with open(p, "r") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 2 # Header + 1 row
        assert reader[-1] == ["1", "42.0"]
        
def test_summary_generation(exp_manager):
    exp_manager.write_summary({"status": "SUCCESS"})
    p = exp_manager.current_exp_dir / "summary.json"
    assert p.exists()
    with open(p, "r") as f:
        s = json.load(f)
    assert s["status"] == "SUCCESS"
