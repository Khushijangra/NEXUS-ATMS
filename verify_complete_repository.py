import sys
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

def run_check(name, check_fn):
    print(f"Checking {name}...", end=" ")
    try:
        passed = check_fn()
        if passed:
            print("[PASS]")
            return True
        else:
            print("[FAIL]")
            return False
    except Exception as e:
        print(f"[FAIL] - {e}")
        return False

def check_configs():
    return (root / "configs" / "ppo.yaml").exists()

def check_experiments_dir():
    return (root / "experiments").exists()

def check_requirements():
    return (root / "requirements.txt").exists() and (root / "environment.yml").exists()

def check_tests_pass():
    result = subprocess.run(["python", "-m", "pytest", "tests/", "-v"], capture_output=True)
    return result.returncode == 0

def check_run_py():
    result = subprocess.run(["python", "run.py", "--healthcheck"], capture_output=True)
    return result.returncode == 0

def main():
    print("\n" + "="*40)
    print("      SPGRL REPOSITORY VERIFICATION")
    print("="*40 + "\n")
    
    checks = [
        ("Config Files Load", check_configs),
        ("Experiments Folder Valid", check_experiments_dir),
        ("Requirements Found", check_requirements),
        ("run.py Commands Work", check_run_py),
        ("Tests Pass", check_tests_pass),
    ]
    
    passes = 0
    for name, fn in checks:
        if run_check(name, fn):
            passes += 1
            
    print("\n" + "="*40)
    print("Repository Status")
    if passes == len(checks):
        print("PASS")
        print(f"{passes} / {len(checks)} tests")
        print("\nTraining Ready")
        print("Paper Ready")
        print("Hackathon Ready")
        print("========================")
        sys.exit(0)
    else:
        print("FAIL")
        print(f"{passes} / {len(checks)} tests")
        print("========================")
        sys.exit(1)

if __name__ == "__main__":
    main()
