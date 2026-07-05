import os
import re
from pathlib import Path

def inject_figure():
    project_root = Path(__file__).resolve().parents[0]
    master_path = project_root / "v2" / "final_dissertation_manuscript" / "SPGRL_IEEE_FINAL_MANUSCRIPT.tex"
    
    with open(master_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Add TikZ packages to preamble
    if r"\usepackage{tikz}" not in content:
        content = content.replace(
            r"\usepackage{xcolor}",
            r"\usepackage{xcolor}" + "\n" + r"\usepackage{tikz}" + "\n" + r"\usetikzlibrary{positioning, shapes.geometric, arrows.meta}"
        )

    # 2. Extract ASCII block
    
    tikz_block = r"""
\begin{figure*}[htbp]
\centering
\resizebox{0.9\textwidth}{!}{%
\begin{tikzpicture}[
    node distance=1.5cm and 0.8cm,
    stream/.style={rectangle, draw=black, thick, fill=gray!10, text width=2.4cm, align=center, rounded corners, minimum height=0.8cm},
    feat/.style={rectangle, draw=black, thick, fill=orange!10, text width=2.4cm, align=center, rounded corners, minimum height=0.8cm},
    out/.style={rectangle, draw=black, thick, fill=blue!10, text width=2.4cm, align=center, rounded corners, minimum height=0.8cm},
    core/.style={rectangle, draw=black, thick, fill=purple!10, text width=3.5cm, align=center, rounded corners, minimum height=1.0cm},
    final/.style={rectangle, draw=black, thick, fill=red!10, text width=3.5cm, align=center, rounded corners, minimum height=1.0cm},
    arrow/.style={->, thick, >=stealth}
]

% Semantic Stream
\node[stream] (video) {Raw Traffic Video};
\node[stream, below=0.5cm of video] (videomae) {Offline VideoMAE};
\node[feat, below=0.5cm of videomae] (features) {768-D Features};
\node[stream, below=0.5cm of features] (mulde) {MULDE + GMM};
\node[out, below=0.5cm of mulde] (sem_anom) {Semantic Anomaly ($A_s$)};

\draw[arrow] (video) -- (videomae);
\draw[arrow] (videomae) -- (features);
\draw[arrow] (features) -- (mulde);
\draw[arrow] (mulde) -- (sem_anom);

% Behavioral Stream
\node[stream, right=0.4cm of video] (yolo_in) {Camera Feed};
\node[stream, below=1.3cm of yolo_in] (yolo) {YOLO + DeepSORT};
\node[out, below=1.7cm of yolo] (beh_anom) {Behavioral Anomaly ($A_b$)};
\draw[arrow] (yolo_in) -- (yolo);
\draw[arrow] (yolo) -- (beh_anom);

% Temporal Stream
\node[stream, right=0.4cm of yolo_in] (lstm_in) {Historical Traffic};
\node[stream, below=1.3cm of lstm_in] (lstm) {LSTM};
\node[out, below=1.7cm of lstm] (lstm_out) {Prediction ($F_t, C_f$)};
\draw[arrow] (lstm_in) -- (lstm);
\draw[arrow] (lstm) -- (lstm_out);

% Spatial Stream
\node[stream, right=0.4cm of lstm_in] (gnn_in) {Neighbor Graph};
\node[stream, below=1.3cm of gnn_in] (gnn) {GNN};
\node[out, below=1.7cm of gnn] (gnn_out) {Spatial ($G_t$)};
\draw[arrow] (gnn_in) -- (gnn);
\draw[arrow] (gnn) -- (gnn_out);

% Sustainability Stream
\node[stream, right=0.4cm of gnn_in] (carbon_in) {Emissions Data};
\node[stream, below=1.3cm of carbon_in] (carbon) {Carbon Engine};
\node[out, below=1.7cm of carbon] (carbon_out) {Carbon ($C_t$)};
\draw[arrow] (carbon_in) -- (carbon);
\draw[arrow] (carbon) -- (carbon_out);

% Emergency Stream
\node[stream, right=0.4cm of carbon_in] (em_in) {V2X / Audio};
\node[stream, below=1.3cm of em_in] (em) {Emergency Routing};
\node[out, below=1.7cm of em] (em_out) {Emergency ($E_t$)};
\draw[arrow] (em_in) -- (em);
\draw[arrow] (em) -- (em_out);

% Unified State
\node[core, below=2.5cm of lstm_out, xshift=1.4cm] (unified) {Unified State ($Z_t$)};

\draw[arrow] (sem_anom.south) -- (unified.north west);
\draw[arrow] (beh_anom.south) -- (unified.north west);
\draw[arrow] (lstm_out.south) -- (unified.north);
\draw[arrow] (gnn_out.south) -- (unified.north east);
\draw[arrow] (carbon_out.south) -- (unified.north east);
\draw[arrow] (em_out.south) -- (unified.east);

% RL and Control
\node[core, below=0.8cm of unified] (mappo) {MAPPO (CTDE)};
\node[core, below=0.8cm of mappo] (safety) {Safety Shield};
\node[final, below=0.8cm of safety] (control) {Traffic Signal Control};

\draw[arrow] (unified) -- (mappo);
\draw[arrow] (mappo) -- (safety);
\draw[arrow] (safety) -- (control);

\end{tikzpicture}%
}
\caption{The hierarchical 6-stream Semantic Predictive Graph Reinforcement Learning (SPGRL) architecture. Unconstrained visual, predictive, and structural data flow concurrently into the Unified State ($Z_t$) for downstream MAPPO evaluation and deterministic Safety Shield interception.}
\label{fig:spgrl_pipeline}
\end{figure*}
"""

    pattern = re.compile(r"\\begin\{verbatim\}\s*Raw Traffic Video[\s\S]*?\\end\{verbatim\}")
    match = pattern.search(content)
    if match:
        content = content[:match.start()] + tikz_block.strip() + content[match.end():]

    with open(master_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Injected TikZ figure successfully.")

if __name__ == "__main__":
    inject_figure()
