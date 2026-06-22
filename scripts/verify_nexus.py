import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Asus\OneDrive\Desktop\projects\nexus-atms")

def run_cmd(cmd, cwd):
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30)
        return "WORKING" if result.returncode == 0 else "BROKEN"
    except Exception as e:
        return "BROKEN"

def verify_nexus():
    status = {}

    # Test train.py imports and help
    status['train.py'] = run_cmd([sys.executable, "train.py", "--help"], cwd=ROOT)
    
    # Test evaluate.py imports and help
    status['evaluate.py'] = run_cmd([sys.executable, "evaluate.py", "--help"], cwd=ROOT)
    
    # Test core modules by importing them
    import_test_cmd = [
        sys.executable, "-c",
        "import control.traffic_env; import ai.rl.ppo_agent; import ai.rl.d3qn_agent; import modules.emergency.corridor; import ai.analytics.carbon"
    ]
    status['Core Modules (TrafficEnv, PPO, D3QN)'] = run_cmd(import_test_cmd, cwd=ROOT)
    
    # Generate report
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    with open(docs_dir / "NEXUS_RUNTIME_STATUS.md", "w") as f:
        f.write("# NEXUS-ATMS RUNTIME STATUS\n\n")
        f.write("| Subsystem | Status |\n|---|---|\n")
        for k, v in status.items():
            f.write(f"| {k} | {v} |\n")
            
    print("NEXUS-ATMS validation complete. Reports generated in docs/.")

if __name__ == "__main__":
    # Ensure SUMO_HOME is set or at least we don't crash on simple import
    os.environ['PYTHONPATH'] = str(ROOT)
    verify_nexus()
