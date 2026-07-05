import os
import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[0]

def create_file(path, content):
    p = project_root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content)

def freeze_v2():
    print("Freezing v2...")
    create_file("v2/FROZEN_RELEASE.md", "# V2 FROZEN RELEASE\nArchitecture is formally locked.")
    create_file("v2/RELEASE_TAG_v2.0.md", "# TAG: v2.0\nCodebase locked for publication.")
    checksums = {"architecture_frozen": True, "checksum": "abc123frozen"}
    create_file("v2/ARCHITECTURE_CHECKSUMS.json", json.dumps(checksums, indent=4))

def create_v3_structure():
    print("Creating V3_HPC_EXPERIMENTS structure...")
    dirs = [
        "semantic", "mappo", "gnn", "joint", "emergency", "statistics",
        "docker", "slurm", "logs", "checkpoints", "telemetry", "figures", "papers"
    ]
    for d in dirs:
        (project_root / "V3_HPC_EXPERIMENTS" / d).mkdir(parents=True, exist_ok=True)

def generate_docker():
    print("Generating Docker stack...")
    dockerfile = """FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip curl software-properties-common
RUN add-apt-repository ppa:sumo/stable && apt-get update && apt-get install -y sumo sumo-tools sumo-doc
RUN python3.11 -m pip install torch==2.5 torchvision torchaudio
RUN python3.11 -m pip install torch-geometric stable-baselines3 traci opencv-python transformers networkx scipy pandas numpy matplotlib seaborn
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
"""
    create_file("V3_HPC_EXPERIMENTS/docker/Dockerfile", dockerfile)
    
    compose = """version: '3.8'
services:
  hpc-node:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ../:/workspace
    working_dir: /workspace
"""
    create_file("V3_HPC_EXPERIMENTS/docker/docker-compose.yml", compose)
    
    entry = "#!/bin/bash\nexec \"$@\"\n"
    create_file("V3_HPC_EXPERIMENTS/docker/entrypoint.sh", entry)
    
    env = """name: v3_hpc
channels:
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.11
  - pytorch=2.5
  - torchvision
  - torchaudio
  - pytorch-cuda=12.4
"""
    create_file("V3_HPC_EXPERIMENTS/docker/environment.yml", env)

def generate_slurm():
    print("Generating SLURM scripts...")
    base_slurm = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu

echo "Starting job on $(hostname)"
python {script}
"""
    scripts = {
        "run_semantic.slurm": "semantic/run_videomae.py",
        "run_gnn.slurm": "gnn/run_scale.py",
        "run_mappo.slurm": "mappo/run_10000_episodes.py",
        "run_joint.slurm": "joint/run_optimization.py",
        "run_emergency.slurm": "emergency/run_routing.py",
        "run_statistics.slurm": "statistics/run_tests.py"
    }
    
    for filename, py_script in scripts.items():
        create_file(f"V3_HPC_EXPERIMENTS/slurm/{filename}", base_slurm.format(script=py_script))
        # Create empty stub
        create_file(f"V3_HPC_EXPERIMENTS/{py_script}", f"# TODO: Implement {py_script}")

def generate_registries():
    print("Generating registries...")
    create_file("V3_HPC_EXPERIMENTS/MANIFEST.yaml", "version: 1.0\ndescription: Phase III HPC Experimental Campaign\n")
    create_file("V3_HPC_EXPERIMENTS/DATASET_REGISTRY.json", '{"BDD100K": "path/to/bdd", "Cityscapes": "path/to/cs"}')
    create_file("V3_HPC_EXPERIMENTS/CHECKPOINT_REGISTRY.json", '{}')
    create_file("V3_HPC_EXPERIMENTS/SEED_REGISTRY.json", '{"seeds": [42, 123, 999, 5050, 10000]}')
    create_file("V3_HPC_EXPERIMENTS/EXPERIMENT_REGISTRY.json", '{}')
    
    hw = {
        "target_gpu": "NVIDIA A100 80GB",
        "fallback_gpu": ["H100", "A6000", "V100"],
        "cpu_cores": 64,
        "ram_gb": 512,
        "storage": "2 TB SSD"
    }
    create_file("V3_HPC_EXPERIMENTS/HARDWARE_PROFILE.json", json.dumps(hw, indent=4))

if __name__ == "__main__":
    freeze_v2()
    create_v3_structure()
    generate_docker()
    generate_slurm()
    generate_registries()
    print("PHASE III HPC SCAFFOLD COMPLETE.")
