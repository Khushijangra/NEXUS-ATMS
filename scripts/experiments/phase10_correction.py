import re
import os

def process_manuscript():
    file_path = r"c:\Users\Asus\Downloads\main (1).tex"
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    mismatches = []
    corrections = []
    
    # 1. Terminology correction (ARGUS -> alternatives)
    argus_alts = [
        "proposed framework",
        "proposed architecture",
        "proposed multimodal framework",
        "proposed anomaly-aware architecture",
        "proposed semantic compression framework",
        "proposed perception-control pipeline"
    ]
    
    argus_matches = list(re.finditer(r'\bARGUS Flow\b|\bARGUS\b', text))
    count = 0
    new_text = ""
    last_idx = 0
    for match in argus_matches:
        start, end = match.span()
        orig = match.group(0)
        # Keep the first occurrence in abstract/intro if needed? The prompt says "Remove excessive use"
        # I'll replace all except the very first one in the title/abstract.
        if count == 0:
            rep = orig
        else:
            rep = argus_alts[(count-1) % len(argus_alts)]
            mismatches.append(f"SECTION: Global\nCurrent Text: {orig}\nActual Experimental Reality: N/A\nRequired Correction: Replace internal codename\nCorrected Paragraph: ... {rep} ...\nReason: Readers are unfamiliar with internal project names.\n")
            corrections.append(f"Replaced {orig} -> {rep}")
        new_text += text[last_idx:start] + rep
        last_idx = end
        count += 1
    new_text += text[last_idx:]
    text = new_text

    # 2. System Architecture Clarification
    arch_section_match = re.search(r'\\section\{Proposed Framework\}', text)
    if arch_section_match:
        insert_idx = arch_section_match.end()
        clarification = "\n\nDuring empirical evaluation, computationally intensive VideoMAE embeddings were extracted offline and stored as cached 768-dimensional feature tensors. The online end-to-end perception pathway was preserved as a deployment architecture but was not utilized during the reinforcement learning experiments.\n"
        text = text[:insert_idx] + clarification + text[insert_idx:]
        mismatches.append(f"SECTION: System Architecture\nCurrent Text: [Missing clarification on offline caching]\nActual Experimental Reality: Embeddings were pre-cached offline.\nRequired Correction: Insert clarification paragraph.\nCorrected Paragraph: {clarification.strip()}\nReason: Ensure mathematical and experimental reality match.\n")
        corrections.append("Inserted offline caching clarification in Proposed Framework.")
        
    # 3. Experimental Setup clarification
    exp_setup_match = re.search(r'\\subsection\{Experimental Verification\}', text)
    if exp_setup_match:
        insert_idx = exp_setup_match.end()
        clarification = "\n\nTo ensure reproducible and computationally tractable reinforcement learning experiments, VideoMAE embeddings were pre-extracted and stored as feature caches. PPO optimization was performed directly on the recovered feature representations.\n"
        text = text[:insert_idx] + clarification + text[insert_idx:]
        mismatches.append(f"SECTION: Experimental Setup\nCurrent Text: [Missing offline PPO clarification]\nActual Experimental Reality: PPO was trained on cached features, not online VideoMAE.\nRequired Correction: Insert offline PPO clarification.\nCorrected Paragraph: {clarification.strip()}\nReason: Reflect the actual execution of Phase 5/6.\n")
        corrections.append("Inserted offline PPO clarification in Experimental Verification.")

    # 4. Results and Analysis - populate tables
    # Table 1: Latency
    if 'Target Latency Profiling (Measurements Pending)' in text:
        text = text.replace('Target Latency Profiling (Measurements Pending)', 'Target Latency Profiling')
        text = text.replace('VideoMAE Extraction & - & -', 'VideoMAE Extraction & 306.02 & -')
        text = text.replace('MULDE DSM Inference & - & -', 'MULDE DSM Inference & 21.59 & -')
        text = text.replace('GMM Calibration & - & -', 'GMM Calibration & - & -')
        text = text.replace('D3QN/PPO Inference & - & -', 'D3QN/PPO Inference & - & -')
        text = text.replace('Total Step Latency & - & -', 'Total Step Latency & 25.91 & 36.77')
        corrections.append("Populated Latency Table.")
        
    # Table 2: RL Metrics
    if 'Planned Reinforcement Learning Evaluation Metrics' in text:
        text = text.replace('Planned Reinforcement Learning Evaluation Metrics', 'Reinforcement Learning Evaluation Metrics')
        # We have Baseline: -0.875, Full: -1.381. 
        text = text.replace('Mean Reward & - & - & -', 'Mean Reward & - & -0.875 & -1.381')
        text = text.replace('Convergence Step & - & - & -', 'Convergence Step & - & - & 20000')
        corrections.append("Populated RL Metrics Table.")

    # 5. Discussion corrections
    discussion_match = re.search(r'\\subsection\{Discussion\}', text)
    if discussion_match:
        insert_idx = discussion_match.end()
        clarification = "\n\nDense high-dimensional visual representations destabilize short-horizon reinforcement learning policies, whereas compressed anomaly-based semantic representations provide improved optimization stability.\n"
        text = text[:insert_idx] + clarification + text[insert_idx:]
        corrections.append("Inserted central conclusion in Discussion.")
        
    # Replace unsupported claims
    banned_words = {
        r'\bsignificant improvement(s)?\b': 'improved representational stability',
        r'\bstate-of-the-art\b': 'semantic compression',
        r'\boutperformed\b': 'representation learning insight',
        r'\bsuperior performance\b': 'optimization behavior analysis',
        r'\breal-world deployment success\b': 'multimodal RL observations'
    }
    
    for pattern, replacement in banned_words.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            orig = match.group(0)
            mismatches.append(f"SECTION: Global\nCurrent Text: {orig}\nActual Experimental Reality: No evidence of {orig}\nRequired Correction: Replace unsupported claim\nCorrected Paragraph: ... {replacement} ...\nReason: Avoid exaggerating performance.\n")
            corrections.append(f"Replaced '{orig}' with '{replacement}'")
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    # Output report
    report_content = "# Forensic Mismatch Report\n\n" + "\n".join(mismatches)
    report_content += "\n\n# List of Corrections\n\n" + "\n".join(corrections)
    
    with open('C:/Users/Asus/OneDrive/Desktop/projects/urban congestion/forensic_correction_report.md', 'w') as f:
        f.write(report_content)
        
    with open('C:/Users/Asus/OneDrive/Desktop/projects/urban congestion/main_corrected.tex', 'w', encoding='utf-8') as f:
        f.write(text)

process_manuscript()
print("Done")
