import os
import shutil
import sys
from pathlib import Path

root = Path(r"D:\Research\urban_congestion")
argus_dir = root / "argus_stream_extracted" / "argus stream A"
anomaly_dir = root / "modules" / "anomaly"

# Move argus to modules/anomaly
if argus_dir.exists():
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    for item in argus_dir.iterdir():
        shutil.move(str(item), str(anomaly_dir / item.name))
    
    # Delete argus_stream_extracted
    shutil.rmtree(str(root / "argus_stream_extracted"))

# Update extract_ua_detrac_features.py
extract_script = root / "scripts" / "extract_ua_detrac_features.py"
if extract_script.exists():
    content = extract_script.read_text()
    content = content.replace('"argus_stream_extracted" / "argus stream A"', '"modules" / "anomaly"')
    content = content.replace('ARGUS_STREAM_A', 'MODULES_ANOMALY')
    extract_script.write_text(content)

# Update train.py
train_script = anomaly_dir / "scripts" / "train.py"
if train_script.exists():
    content = train_script.read_text()
    # Path(__file__).resolve().parent.parent -> Path(__file__).resolve().parent.parent.parent
    content = content.replace('Path(__file__).resolve().parent.parent', 'Path(__file__).resolve().parent.parent.parent')
    train_script.write_text(content)

# Update generate_ua_detrac_metadata.py
gen_script = anomaly_dir / "scripts" / "generate_ua_detrac_metadata.py"
if gen_script.exists():
    # It already handles images/train and images/val correctly.
    pass

print("Restructure complete")
