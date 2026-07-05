import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[0]
out_dir = project_root / "v2" / "final_dissertation_manuscript"
out_dir.mkdir(parents=True, exist_ok=True)

def write_file(name, content):
    with open(out_dir / name, 'w', encoding='utf-8') as f:
        f.write(content)

def generate_paper():
    tex = r"""\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2023.0322000}

\title{A Unified Semantic Predictive Graph Reinforcement Learning Framework for Sustainable Urban Traffic Signal Control}
\author{\uppercase{First A. Author}\authorrefmark{1},
\uppercase{Second B. Author\authorrefmark{2}, and Third C. Author\authorrefmark{3}}}
\address[1]{Department of Computer Science, University of Technology, City, Country}
\tfootnote{This work was supported in part by the National Science Foundation under Grant XXXX.}

\markboth
{Author \headeretal: A Unified Semantic Predictive Graph Reinforcement Learning Framework}
{Author \headeretal: A Unified Semantic Predictive Graph Reinforcement Learning Framework}

\corresp{Corresponding author: First A. Author (e-mail: author@university.edu).}

\begin{abstract}
Urban traffic congestion remains a critical challenge, primarily because current Traffic Signal Control (TSC) systems rely on reactive heuristics that fail to understand spatial semantics, forecast future trajectories, coordinate across graph topologies, optimize for carbon emissions, or handle emergency routing. These siloed approaches struggle to manage unpredictable anomaly events and fail to scale effectively in multi-intersection environments.

To overcome these limitations, we propose the Semantic Predictive Graph Reinforcement Learning (SPGRL) framework. This multimodal architecture integrates VideoMAE, MULDE, and GMM for semantic anomaly detection ($A_s$), YOLO and DeepSORT for behavioral tracking ($A_b$), and LSTM for trajectory forecasting ($F_t, C_f$). Simultaneously, a Graph Neural Network (GNN) coordinates the road topology ($G_t$), while a Carbon Engine bounds emissions ($C_t$) and an Emergency Routing protocol guarantees safety ($E_t$).

These multimodal streams are fused into a highly dense unified state $Z_t = [G_t, A_s, A_b, F_t, C_f, C_t, E_t]$, forming the basis of a Multi-Agent Proximal Policy Optimization (MAPPO) framework fortified by a deterministic Safety Shield.

We explicitly define the architectural contributions and propose a comprehensive experimental protocol leveraging SUMO, BDD100K, and Cityscapes across 64 intersections for 10,000 episodes over 5 random seeds on an NVIDIA A100 cluster. Large-scale empirical validation will be conducted using the Phase III HPC pipeline.
\end{abstract}

\begin{keywords}
Reinforcement Learning, Intelligent Transportation Systems, Graph Neural Networks, Multi-Agent Systems, VideoMAE, Anomaly Detection.
\end{keywords}

\titlepgskip=-15pt
\maketitle

\section{Introduction}
\label{sec:introduction}
Current intelligent transportation systems suffer from six fundamental limitations that prevent true autonomy in urban traffic networks:
1) \textbf{Reactive control:} Systems only respond to current queues rather than anticipating incoming congestion.
2) \textbf{No anomaly understanding:} They treat accidents and debris identically to normal vehicle density.
3) \textbf{No predictive forecasting:} They lack the temporal horizon to project future state variables.
4) \textbf{No graph coordination:} Intersections act selfishly rather than cooperatively passing hidden states.
5) \textbf{No carbon optimization:} They strictly optimize throughput, ignoring severe environmental costs.
6) \textbf{No emergency guarantees:} They leave ambulances vulnerable to stochastic RL exploration.

To address these, we introduce the Semantic Predictive Graph Reinforcement Learning (SPGRL) framework. The major contributions of this work are:
\begin{enumerate}
    \item We propose the first Semantic Predictive Graph Reinforcement Learning (SPGRL) framework for urban traffic control.
    \item We introduce a dual-stream anomaly architecture combining semantic video perception and behavioral trajectory analysis.
    \item We develop a VideoMAE-MULDE-GMM semantic anomaly pipeline.
    \item We formulate a behavioral anomaly engine using YOLO and DeepSORT trajectory statistics.
    \item We integrate LSTM forecasting directly into the RL state space.
    \item We develop a graph-based MAPPO coordination mechanism using CTDE.
    \item We propose a carbon-aware optimization strategy.
    \item We introduce a deterministic emergency Safety Shield.
    \item We formulate a unified multimodal state representation capable of joint optimization without catastrophic interference.
\end{enumerate}

\section{Related Work}
\label{sec:related_work}
\subsection{RL Traffic Signal Control}
\cite{placeholder}
\subsection{Predictive RL}
\cite{placeholder}
\subsection{Graph RL}
\cite{placeholder}
\subsection{Multi-Agent RL}
\cite{placeholder}
\subsection{Traffic Forecasting}
\cite{placeholder}
\subsection{Semantic Traffic Understanding}
\cite{placeholder}
\subsection{VideoMAE}
\cite{placeholder}
\subsection{Anomaly Detection}
\cite{placeholder}
\subsection{Carbon Optimization}
\cite{placeholder}
\subsection{Emergency Routing}
\cite{placeholder}
\subsection{Safety Shield Methods}
\cite{placeholder}

\section{Methodology}
\label{sec:methodology}

\subsection{Overall SPGRL Architecture}
The unified architecture bridges Raw Video, Trajectory Streams, Traffic History, Road Graphs, Carbon Engines, and Emergency Engines into a single multimodal processing pipeline.

\subsection{Semantic Anomaly Module}
Raw video features are extracted via VideoMAE into a 768-D embedding, modeled via MULDE and GMM to yield the semantic anomaly:
$$ A_s = -\log P(x) $$

\subsection{Behavioral Anomaly Module}
Trajectories extracted from YOLO and DeepSORT yield velocity, acceleration, jerk, entropy, and wrong-way telemetry:
$$ A_b = 0.30z_v + 0.25z_a + 0.20j_t + 0.15H + 0.10W $$

The final fusion integrates both streams:
$$ A_t = \alpha A_s + (1-\alpha) A_b $$

\subsection{Traffic Prediction Module}
Historical traffic is fed into an LSTM to generate predictions $F_t$. The forecast confidence is:
$$ C_f = 1 - \frac{\sigma(F_t)}{\max(\sigma)} $$

\subsection{Graph Representation Module}
The road graph is mapped through GCN/GAT layers to establish intersection neighborhood representations $G_t$.

\subsection{Carbon Engine}
Emissions are modeled via vehicle kinetics:
$$ C_t = \sum_i CO_2(v_i, a_i) $$

\subsection{Emergency Routing Module}
The Safety Shield engine generates optimal paths yielding $E_t$.

\subsection{Unified State Construction}
The individual modalities are concatenated into a high-dimensional state vector:
$$ Z_t = [G_t, A_s, A_b, F_t, C_f, C_t, E_t] $$

\subsection{MAPPO CTDE}
The Centralized Training with Decentralized Execution mechanism governs the Actor and Critic networks, optimized by:
$$ L_{PPO} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t) A_t)] $$

\subsection{Joint Optimization}
We backpropagate the unified joint loss:
$$ L_{total} = L_{PPO} + \lambda_1 L_{LSTM} + \lambda_2 L_{GNN} $$
Cosine similarity prevents catastrophic interference:
$$ \text{Cos}(\theta) = \frac{g_1 \cdot g_2}{|g_1| |g_2|} $$

\subsection{Safety Shield}
An absolute emergency override mechanism deterministically assumes control when emergency vectors dictate collision avoidance priorities.

\section{Optimization Objective}
\label{sec:optimization_objective}
The final reward is formulated as:
$$ R_t = w_1 R_{traffic} - w_2 C_t - w_3 A_t + w_4 E_t + w_5 C_f $$

\section{Computational Complexity}
\label{sec:complexity}
\begin{table}[htbp]
\centering
\caption{Computational Complexity}
\begin{tabular}{|c|c|}
\hline
\textbf{Module} & \textbf{Complexity} \\
\hline
Behavioral & $\mathcal{O}(N)$ \\
Emergency & $\mathcal{O}(E+V\log V)$ \\
LSTM & $\mathcal{O}(WH)$ \\
GNN & $\mathcal{O}(V+E)$ \\
PPO & $\mathcal{O}(|Z||A|)$ \\
Safety Shield & $\mathcal{O}(E+V\log V)$ \\
Unified State & $\mathcal{O}(|Z|)$ \\
\hline
\end{tabular}
\end{table}

\section{Experimental Setup}
\label{sec:experimental_setup}
The experimental protocol utilizes SUMO, BDD100K, and Cityscapes. We test grid topologies of 1x1, 2x2, 4x4, 8x8, up to 64 intersections. Training executes over 10,000 episodes across 5 random seeds on NVIDIA A100 clusters (64 CPU, 512GB RAM) using Docker, SLURM, and PyTorch (CUDA).

\section{Results}
\label{sec:results}
\subsection{Semantic anomaly evaluation}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{Behavioral anomaly evaluation}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{Fusion coefficient ablation}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{LSTM forecasting}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{Forecast confidence calibration}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{Carbon Pareto analysis}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{Graph scalability}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{MAPPO convergence}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{Joint optimization stability}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{Gradient similarity analysis}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{Emergency routing evaluation}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{Safety shield evaluation}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
\subsection{System-level SPGRL evaluation}
[PLACEHOLDER AWAITING HPC V3 EXECUTION]

\section{Discussion}
\label{sec:discussion}
We expect the SPGRL framework to demonstrate significant advantages in bounding latency while navigating anomalies. The scaling behavior is hypothesized to remain robust up to 64 intersections, though generalization to unstructured road networks remains an active area of investigation.

\section{Limitations}
\label{sec:limitations}
The architecture faces strict HPC requirements; the memory cost of VideoMAE precludes straightforward edge inference deployment. Additionally, centralized critic communication assumptions may fail under realistic bandwidth constraints. Dataset bias and deployment complexity remain formidable challenges.

\section{Future Work}
\label{sec:future_work}
Future work focuses on transitioning to Spatial-Temporal GNNs (STGNN) and Vision Transformers. We aim to apply knowledge distillation and Federated MAPPO to enable localized Jetson deployment and integration into live smart city digital twins.

\section{Conclusion}
\label{sec:conclusion}
This work proposes a unified Semantic Predictive Graph Reinforcement Learning architecture that integrates semantic perception, behavioral analysis, predictive forecasting, graph reasoning, carbon optimization, emergency routing, and multi-agent reinforcement learning into a mathematically consistent traffic control framework.

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
"""
    write_file("paper_complete_revised.tex", tex)

def generate_equations():
    eqs = r"""% Updated Equations for SPGRL

% Semantic anomaly
$$ A_s = -\log P(x) $$

% Behavioral anomaly
$$ A_b = 0.30z_v + 0.25z_a + 0.20j_t + 0.15H + 0.10W $$

% Fusion
$$ A_t = \alpha A_s + (1-\alpha) A_b $$

% Forecast confidence
$$ C_f = 1 - \frac{\sigma(F_t)}{\max(\sigma)} $$

% Carbon
$$ C_t = \sum_i CO_2(v_i, a_i) $$

% Unified state
$$ Z_t = [G_t, A_s, A_b, F_t, C_f, C_t, E_t] $$

% MAPPO objective
$$ L_{PPO} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t) A_t)] $$

% Joint optimization
$$ L_{total} = L_{PPO} + \lambda_1 L_{LSTM} + \lambda_2 L_{GNN} $$

% Gradient similarity
$$ \text{Cos}(\theta) = \frac{g_1 \cdot g_2}{|g_1| |g_2|} $$

% Final reward
$$ R_t = w_1 R_{traffic} - w_2 C_t - w_3 A_t + w_4 E_t + w_5 C_f $$
"""
    write_file("updated_equations.tex", eqs)

def generate_figures():
    figs = r"""% Updated Figures Placeholders
% Fig1 SPGRL architecture
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Fig2 Semantic anomaly pipeline
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Fig3 Behavioral anomaly pipeline
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Fig4 LSTM forecasting
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Fig5 Carbon Pareto
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Fig6 GNN scaling
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Fig7 MAPPO convergence
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Fig8 Gradient similarity
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Fig9 Emergency routing
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Fig10 Unified Zt architecture
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
"""
    write_file("updated_figures.tex", figs)

def generate_tables():
    tabs = r"""% Updated Tables Placeholders
% Table1 Symbols
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Table2 Hyperparameters
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Table3 Complexity
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Table4 Ablation studies
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Table5 Statistical tests
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
% Table6 Hardware profile
[PLACEHOLDER AWAITING HPC V3 EXECUTION]
"""
    write_file("updated_tables.tex", tabs)

if __name__ == "__main__":
    generate_paper()
    generate_equations()
    generate_figures()
    generate_tables()
    print("FINAL MANUSCRIPT RECONSTRUCTION COMPLETE.")
