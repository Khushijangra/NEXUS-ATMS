import os

dir1 = 'C:/Users/Asus/OneDrive/Desktop/projects/nexus-atms'
dir2 = 'C:/Users/Asus/OneDrive/Desktop/projects/argus-flow'
out_file = 'C:/Users/Asus/OneDrive/Desktop/projects/urban congestion/forensics/SHARED_CORE.md'

common = []
for root, _, files in os.walk(dir1):
    if '.git' in root:
        continue
    for f in files:
        rel_path = os.path.relpath(os.path.join(root, f), dir1)
        path2 = os.path.join(dir2, rel_path)
        if os.path.exists(path2):
            common.append(rel_path.replace('\\', '/'))

with open(out_file, 'w', encoding='utf-8') as f:
    f.write('# SHARED CORE COMPONENTS\n\n')
    f.write('The following files exist in both the `nexus-atms` (Base Platform) and `argus-flow` (Incident Intelligence) repositories. These files represent the foundational bridge where ArgusFlow depends on NEXUS architecture.\n\n')
    for c in sorted(common):
        f.write(f'- `{c}`\n')
print(f"Done. Found {len(common)} shared files.")

