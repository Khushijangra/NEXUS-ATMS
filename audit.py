import os
import json
from pathlib import Path

root = Path(r"D:\Research\urban_congestion")

target_dirs = {
    'configs', 'data', 'modules', 'networks', 'services', 'scripts', 'tests', 'control', 'core', 'config', 'archive'
}
target_files = {
    'train.py', 'evaluate.py', 'run_demo.py', 'run_digital_twin.py',
    'requirements.txt', 'README.md', 'LICENSE', '.gitignore'
}

to_delete_dirs = []
to_delete_files = []

for item in root.iterdir():
    if item.name.startswith('.git') and item.name != '.gitignore':
        continue # keep git
    
    if item.is_dir():
        if item.name not in target_dirs:
            to_delete_dirs.append(item.name)
    elif item.is_file():
        if item.name not in target_files:
            to_delete_files.append(item.name)

# Deep dive into allowed dirs to remove pycache and other garbage
deep_garbage = []
for td in target_dirs:
    td_path = root / td
    if not td_path.exists(): continue
    for root_dir, dirs, files in os.walk(td_path):
        for d in list(dirs):
            if d in ['__pycache__', '.pytest_cache']:
                deep_garbage.append(os.path.join(root_dir, d))
                dirs.remove(d) # don't recurse
        for f in files:
            if f.endswith('.pyc') or f.endswith('.log') or f.endswith('.tmp'):
                deep_garbage.append(os.path.join(root_dir, f))

report = {
    "root_dirs_to_delete": to_delete_dirs,
    "root_files_to_delete": to_delete_files,
    "deep_garbage_count": len(deep_garbage)
}

print(json.dumps(report, indent=2))
