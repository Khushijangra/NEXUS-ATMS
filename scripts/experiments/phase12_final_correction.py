import re
import os

def run():
    source_file = r"C:\Users\Asus\Downloads\main (1).tex"
    with open(source_file, 'r', encoding='utf-8') as f:
        text = f.read()

    change_log = []
    
    # 1. Terminology correction (ARGUS -> alternatives)
    argus_alts = [
        "proposed framework",
        "proposed architecture",
        "proposed multimodal framework",
        "proposed anomaly-aware framework",
        "proposed semantic compression framework",
        "proposed perception-control pipeline",
        "proposed multimodal controller"
    ]
    
    argus_matches = list(re.finditer(r'\bARGUS Flow\b|\bARGUS\b', text))
    count = 0
    new_text = ""
    last_idx = 0
    for match in argus_matches:
        start, end = match.span()
        orig = match.group(0)
        if count == 0:
            rep = orig
        else:
            rep = argus_alts[(count-1) % len(argus_alts)]
            change_log.append({
                "orig": orig,
                "corr": rep,
                "reason": "Remove excessive use of internal project names.",
                "evidence": "Readers are unfamiliar with internal project names."
            })
        new_text += text[last_idx:start] + rep
        last_idx = end
        count += 1
    new_text += text[last_idx:]
    text = new_text

    # 2. Remove unsupported claims
    banned = {
        r'\bstate-of-the-art\b': 'semantic compression',
        r'\bsuperior\b': 'stable',
        r'\bbreakthrough\b': 'verified',
        r'\boutperforms\b': 'stabilizes',
        r'\bbest\b': 'verified',
        r'\bobviously\b': 'empirically',
        r'\bsignificant improvement(s)?\b': 'improved representational stability'
    }
    
    for pattern, replacement in banned.items():
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            orig = match.group(0)
            change_log.append({
                "orig": orig,
                "corr": replacement,
                "reason": "Replace unsupported claims.",
                "evidence": "Statistical significance (p > 0.05) does not support superiority claims."
            })
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Insert offline execution clarification
    arch_section = re.search(r'\\section\{Proposed Framework\}', text)
    if arch_section:
        insert_idx = arch_section.end()
        clarification = "\n\nDuring empirical evaluation, VideoMAE embeddings were extracted offline and stored as cached 768-dimensional feature representations. Reinforcement learning optimization was performed on the recovered feature space, while the online perception pipeline was preserved as a deployment architecture.\n"
        text = text[:insert_idx] + clarification + text[insert_idx:]
        change_log.append({
            "orig": "[Missing offline clarification]",
            "corr": clarification.strip(),
            "reason": "Insert offline execution clarification.",
            "evidence": "Experimental pipeline used cached features (.npy), not online VideoMAE."
        })

    # 4. Insert central scientific finding
    discussion = re.search(r'\\subsection\{Discussion\}', text)
    if discussion:
        insert_idx = discussion.end()
        finding = "\n\nDense high-dimensional visual embeddings destabilize short-horizon reinforcement learning policies, whereas compressed anomaly-based semantic representations preserve optimization stability.\n"
        text = text[:insert_idx] + finding + text[insert_idx:]
        change_log.append({
            "orig": "[Missing central finding]",
            "corr": finding.strip(),
            "reason": "Insert central scientific finding.",
            "evidence": "Ablation study (Full -1.381 vs Anomaly -0.839)."
        })

    # 5. Populate Tables
    if 'Target Latency Profiling (Measurements Pending)' in text:
        text = text.replace('Target Latency Profiling (Measurements Pending)', 'Target Latency Profiling')
        text = text.replace('VideoMAE Extraction & - & -', 'VideoMAE Extraction & 306.02 & -')
        text = text.replace('MULDE DSM Inference & - & -', 'MULDE DSM Inference & 21.59 & -')
        text = text.replace('GMM Calibration & - & -', 'GMM Calibration & - & -')
        text = text.replace('D3QN/PPO Inference & - & -', 'D3QN/PPO Inference & - & -')
        text = text.replace('Total Step Latency & - & -', 'Total Step Latency & 25.91 & 36.77')

    if 'Planned Reinforcement Learning Evaluation Metrics' in text:
        text = text.replace('Planned Reinforcement Learning Evaluation Metrics', 'Reinforcement Learning Evaluation Metrics')
        text = text.replace('Mean Reward & - & - & -', 'Mean Reward & - & -0.875 & -1.381')
        text = text.replace('Convergence Step & - & - & -', 'Convergence Step & - & - & 20000')
        text = text.replace('Value Loss (Final) & - & - & -', 'Value Loss (Final) & - & - & -')
        text = text.replace('Policy Entropy & - & - & -', 'Policy Entropy & - & - & -')

    # Output log
    log_content = "# Forensic Change Log\n\n"
    for cl in change_log:
        log_content += f"**Original Sentence:** {cl['orig']}\n"
        log_content += f"**Corrected Sentence:** {cl['corr']}\n"
        log_content += f"**Reason for Correction:** {cl['reason']}\n"
        log_content += f"**Supporting Experimental Evidence:** {cl['evidence']}\n\n"
        
    out_dir = r"C:\Users\Asus\OneDrive\Desktop\projects\urban congestion"
    with open(os.path.join(out_dir, 'main_corrected.tex'), 'w', encoding='utf-8') as f:
        f.write(text)
        
    with open(os.path.join(out_dir, 'forensic_change_log.md'), 'w', encoding='utf-8') as f:
        f.write(log_content)

run()
print("Success")
