import os
import json

root = r"D:\Research\urban_congestion"
to_delete = []

for root_dir, dirs, files in os.walk(root):
    # Exclude git
    if '.git' in root_dir:
        continue
        
    rel_root = os.path.relpath(root_dir, root)
    if rel_root == '.':
        rel_root = ''
        
    # Mark directories for deletion
    for d in list(dirs):
        full_d = os.path.join(root_dir, d)
        rel_d = os.path.relpath(full_d, root)
        
        # Safe deletion rules
        if d in ['.archive', '.pytest_cache', '.venv', '.vscode', 'audio_cache', 'docs', 'logs', 'models', 'node_modules', 'presentation_assets', 'presentation_diagrams', 'venv']:
            to_delete.append(f"[DIR] {rel_d}")
            dirs.remove(d) # don't recurse
        elif d == '__pycache__':
            to_delete.append(f"[DIR] {rel_d}")
            dirs.remove(d)
            
    # Mark files for deletion
    for f in files:
        full_f = os.path.join(root_dir, f)
        rel_f = os.path.relpath(full_f, root)
        
        if rel_f in ['dummy.pt', 'yolov8n.pt', 'smoke_test.py', 'nexus-start.ps1', 'setup.bat', 'CHANGELOG.md', 'CODE_OF_CONDUCT.md', 'CONTRIBUTING.md', 'SECURITY.md', 'Dockerfile', 'docker-compose.yml', 'Procfile', 'railway.json', 'render.yaml', 'runtime.txt']:
            to_delete.append(f"[FILE] {rel_f}")
        elif f.endswith('.pyc') or f.endswith('.log') or f.endswith('.tmp'):
            to_delete.append(f"[FILE] {rel_f}")
        elif rel_root == 'scripts' and f in ['benchmark_d3qn_suite.py', 'compare_agents.py', 'generate_ai_report.py', 'generate_demo_preview_video.py', 'generate_dti_final_report.py', 'generate_graph_ab_report.py', 'generate_grid_4x4.py', 'generate_nonzero_tables_pdf.py', 'generate_release_notes.py', 'generate_report.py', 'generate_results_pdf.py', 'generate_scenarios.py', 'preprocess_videos.py', 'test_gpu_training.py', 'test_pseudo_live_pipeline.py', 'test_sumo_connection.py', 'ui_acceptance_playwright.py']:
            to_delete.append(f"[FILE] {rel_f}")

with open(r"D:\Research\urban_congestion\plan.json", 'w') as f:
    json.dump(to_delete, f, indent=2)
