import os
from pathlib import Path

def append_results():
    project_root = Path(__file__).resolve().parents[0]
    results_path = project_root / "v2" / "final_dissertation_manuscript" / "RESULTS_SECTION_RECONSTRUCTION.tex"
    master_path = project_root / "v2" / "final_dissertation_manuscript" / "SPGRL_IEEE_FINAL_MANUSCRIPT.tex"
    
    with open(results_path, "r", encoding="utf-8") as f:
        results_content = f.read()
        
    with open(master_path, "r", encoding="utf-8") as f:
        master_content = f.read()
        
    # Remove \end{document}
    master_content = master_content.replace(r"\end{document}", "")
    
    # Append the new section
    master_content += "\n" + results_content + "\n\\end{document}\n"
    
    with open(master_path, "w", encoding="utf-8") as f:
        f.write(master_content)
        
    print("Appended Results to master manuscript.")

if __name__ == "__main__":
    append_results()
