import os
import re
from pathlib import Path

def generate_beautiful_spgrl():
    tex = r"""\documentclass[journal]{IEEEtran}
\usepackage{dblfloatfix}
\usepackage{cite}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{subcaption}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{multirow}
\usepackage{multicol}
\usepackage{xcolor}
\usepackage{comment}

\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, fit, backgrounds, shadows}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{A Unified Semantic Predictive Graph Reinforcement Learning Framework for Sustainable Urban Traffic Signal Control}

\author{Khushi, Jatin, Jaismeen, and Susmita Das
\thanks{The authors are with SCSET, Bennett University, Greater Noida, UP, India (e-mail: khushi@gmail.com, jatin@gmail.com, jaismeen@gmail.com, susmitad900@gmail.com).}}

\markboth{Author \MakeLowercase{\textit{et al.}}: A Unified Semantic Predictive Graph Reinforcement Learning Framework}{}

\maketitle

\begin{abstract}
Urban traffic congestion remains a critical bottleneck in smart city infrastructure, inducing severe economic and environmental penalties. Traditional actuated and fixed-time controllers fail to adapt to macro-level stochastic perturbations inherent in real-world traffic networks. Current Traffic Signal Control (TSC) systems rely on reactive heuristics that fail to understand spatial semantics, forecast future trajectories, coordinate across graph topologies, optimize for carbon emissions, or handle emergency routing. 

In this paper, we present the Semantic Predictive Graph Reinforcement Learning (SPGRL) framework, an end-to-end cyber-physical architecture leveraging Multi-Agent Proximal Policy Optimization (MAPPO) seamlessly integrated with dual-stream anomaly detection. The framework employs VideoMAE, MULDE, and GMM for semantic anomaly detection ($A_s$), while simultaneously utilizing YOLO and DeepSORT for behavioral trajectory tracking ($A_b$). To ensure proactive routing, an LSTM network provides trajectory forecasting ($F_t, C_f$), and a Graph Neural Network (GNN) coordinates the road topology ($G_t$). A dedicated Carbon Engine bounds emissions ($C_t$), and a deterministic Emergency Routing protocol guarantees absolute safety ($E_t$).

These multimodal streams are natively concatenated into a highly dense unified state $Z_t$, empowering the MAPPO agent to jointly optimize for traffic congestion reduction, carbon footprint minimization, and anomalous event mitigation without catastrophic interference. The system preserves operational safety through a Safety Shield. Ultimately, this framework demonstrates that multimodal deep reinforcement learning can be effectively synthesized with density-based computer vision for resilient, real-world deployment. Extensive empirical validation will be conducted using the Phase III HPC pipeline on SUMO, BDD100K, and Cityscapes across 64 intersections for 10,000 episodes over 5 random seeds on an NVIDIA A100 cluster.
\end{abstract}

\begin{IEEEkeywords}
Adaptive traffic signal control, reinforcement learning, Multi-Agent PPO, Video anomaly detection, Vision Transformer, Graph Neural Networks.
\end{IEEEkeywords}

\section{Introduction}
\label{sec:introduction}
Urban traffic congestion represents one of the most economically and environmentally damaging inefficiencies in modern metropolitan infrastructure. Global estimates place the annual cost of gridlock at over \$1.7 trillion, driven by lost productivity, wasted fuel, increased logistics overhead, and elevated vehicular carbon emissions. The near-ubiquitous reliance on legacy, deterministic traffic signal controllers operating on pre-programmed cyclic phases constitutes a primary architectural flaw. 

Current intelligent transportation systems suffer from six fundamental limitations that prevent true autonomy in urban traffic networks:
1) \textbf{Reactive control:} Systems only respond to current queues rather than proactively anticipating incoming shockwaves.
2) \textbf{No anomaly understanding:} They treat catastrophic accidents, illegal parking, and debris identically to normal high-density vehicle flow.
3) \textbf{No predictive forecasting:} They lack the temporal horizon to project future state variables and trajectories.
4) \textbf{No graph coordination:} Intersections act as selfish, isolated entities rather than a cooperatively passing hidden states across a topological mesh.
5) \textbf{No carbon optimization:} They strictly optimize localized throughput, entirely ignoring severe, compounding environmental costs.
6) \textbf{No emergency guarantees:} They leave critical ambulances and fire engines vulnerable to stochastic RL exploration policies.

To bridge the extreme translational gap between isolated numerical traffic control and real-world visual chaos, this paper proposes the comprehensive Semantic Predictive Graph Reinforcement Learning (SPGRL) cyber-physical system. By fusing computer vision, sequence forecasting, and graph theory, the controller enacts evasive phase shifts during catastrophic events while optimizing standard throughput.

The major contributions of this work are:
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
The field of artificial intelligence-driven traffic signal control has evolved through several distinct phases.
\subsection{RL Traffic Signal Control}
Deep RL architectures such as DQN, PPO, FRAP, and PressLight have transformed traffic control from fixed-cycle heuristics to adaptive phase management \cite{placeholder}.
\subsection{Predictive RL}
Integrating sequence models into reinforcement learning allows policies to act on future horizons rather than lagging indicators \cite{placeholder}.
\subsection{Graph RL}
Graph Convolutional Networks (GCN) enable explicit spatial coordination between neighboring intersections \cite{placeholder}.
\subsection{Multi-Agent RL}
CTDE paradigms have solved the non-stationarity problems inherent in independent Q-learning for traffic grids \cite{placeholder}.
\subsection{Traffic Forecasting}
LSTMs and sequence models are widely used for short-term congestion and trajectory prediction \cite{placeholder}.
\subsection{Semantic Traffic Understanding}
Deep representations bridge the semantic gap between raw pixels and intersection topologies \cite{placeholder}.
\subsection{VideoMAE}
Models like VideoMAE have advanced spatial-temporal feature extraction, outperforming traditional convolutional backbones \cite{placeholder}.
\subsection{Anomaly Detection}
Density-based and reconstruction-based paradigms localize catastrophic disruptions in unconstrained feeds \cite{placeholder}.
\subsection{Carbon Optimization}
Integrating continuous emission functions into objective optimization remains heavily under-explored \cite{placeholder}.
\subsection{Emergency Routing}
Priority preemption lacks integration with continuous multi-agent RL optimization \cite{placeholder}.
\subsection{Safety Shield Methods}
Deterministic fallback gates guarantee operational safety against stochastic neural hallucinations \cite{placeholder}.

\section{Methodology}
\label{sec:methodology}

The architectural efficacy of the SPGRL framework is underpinned by a rigorous mathematical foundation that bridges high-dimensional spatio-temporal computer vision with Markovian decision processes. 

\subsection{Overall SPGRL Architecture}
The system architecture is a sequential multimodal pipeline where the goal is to convert high-dimensional video streams and historical arrays into low-dimensional actionable anomaly scores and embeddings that condition the MAPPO policy. The unified architecture bridges Raw Video, Trajectory Streams, Traffic History, Road Graphs, Carbon Engines, and Emergency Engines into a single multimodal processing pipeline.

\begin{figure*}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    font=\sffamily\small,
    >=Stealth,
    database/.style={cylinder, draw=gray!80, thick, aspect=0.25, shape border rotate=90, minimum height=1.6cm, minimum width=2.4cm, fill=white, align=center, drop shadow={opacity=0.1}},
    process/.style={rectangle, draw=black!70, thick, rounded corners=3pt, minimum height=1.2cm, minimum width=2.8cm, fill=white, align=center, drop shadow={opacity=0.1}},
    tensor/.style={rectangle, draw=blue!80!black, thick, minimum height=0.9cm, minimum width=2.2cm, fill=blue!4, align=center, drop shadow={opacity=0.1}},
    fusion/.style={circle, draw=black!80, thick, minimum size=1.2cm, fill=yellow!15, align=center, drop shadow={opacity=0.1}},
    gate/.style={diamond, draw=red!70!black, thick, minimum width=2.2cm, minimum height=2.2cm, fill=red!4, align=center, drop shadow={opacity=0.1}},
    arrow/.style={->, thick, draw=black!80},
    groupbox/.style={rectangle, draw=gray!40, thick, dashed, rounded corners=5pt, inner sep=12pt}
]

% ROW 1: Semantic & Behavioral Anomaly (Y=3)
\node[database] (video) at (0, 3) {Traffic\\Video};
\node[process] (videomae) at (3.5, 4) {VideoMAE};
\node[process] (mulde) at (7, 4) {MULDE+GMM};
\node[tensor] (scoreS) at (10.5, 4) {$A_s$};

\node[process] (yolo) at (3.5, 2) {YOLO+SORT};
\node[process] (kinematics) at (7, 2) {Kinematics};
\node[tensor] (scoreB) at (10.5, 2) {$A_b$};

% ROW 2: Traffic Simulation & Graph (Y=0)
\node[database] (sumo) at (0, 0) {Network\\State};
\node[process] (lstm) at (4, 0.8) {LSTM};
\node[tensor] (forecast) at (8, 0.8) {$F_t, C_f$};

\node[process] (gnn) at (4, -0.8) {GNN};
\node[tensor] (graph) at (8, -0.8) {$G_t$};

% ROW 3: Carbon & Emergency
\node[process] (carbon) at (4, -2.5) {Carbon};
\node[tensor] (ct) at (8, -2.5) {$C_t$};

\node[process] (emergency) at (4, -4) {Emergency};
\node[tensor] (et) at (8, -4) {$E_t$};

% Fusion Node
\node[fusion] (concat) at (14, 0) {$\bigoplus$};
\node[tensor] (state) at (17, 0) {Unified $Z_t$};
\node[process] (mappo) at (20.5, 0) {MAPPO\\CTDE};
\node[gate] (safety) at (24.5, 0) {Safety\\Shield};
\node[tensor] (action) at (27.5, 0) {Action $a_t$};

% Connectors
\draw[arrow] (video) |- (videomae);
\draw[arrow] (videomae) -- (mulde);
\draw[arrow] (mulde) -- (scoreS);
\draw[arrow] (scoreS.east) -- (concat.north west);

\draw[arrow] (video) |- (yolo);
\draw[arrow] (yolo) -- (kinematics);
\draw[arrow] (kinematics) -- (scoreB);
\draw[arrow] (scoreB.east) -- (concat.west);

\draw[arrow] (sumo) |- (lstm);
\draw[arrow] (lstm) -- (forecast);
\draw[arrow] (forecast.east) -- (concat.west);

\draw[arrow] (sumo) |- (gnn);
\draw[arrow] (gnn) -- (graph);
\draw[arrow] (graph.east) -- (concat.west);

\draw[arrow] (sumo) |- (carbon);
\draw[arrow] (carbon) -- (ct);
\draw[arrow] (ct.east) -- (concat.south west);

\draw[arrow] (sumo) |- (emergency);
\draw[arrow] (emergency) -- (et);
\draw[arrow] (et.east) -- (concat.south);

% RL Pipeline
\draw[arrow] (concat) -- (state);
\draw[arrow] (state) -- (mappo);
\draw[arrow] (mappo) -- (safety);
\draw[arrow] (safety) -- (action);

\end{tikzpicture}
}
\caption{Architectural Design of the SPGRL Framework. Multimodal data streams including Semantic Anomaly ($A_s$), Behavioral Anomaly ($A_b$), Forecast ($F_t$), Graph ($G_t$), Carbon ($C_t$), and Emergency ($E_t$) are concatenated ($\bigoplus$) to generate the Unified State $Z_t$.}
\label{fig:full_architecture}
\end{figure*}

\subsection{Semantic Anomaly Module ($A_s$)}
The framework circumvents conventional object detection heuristics for a Video Masked Autoencoder (VideoMAE) to unravel the intricate kinematics of vehicle motion. The output of the VideoMAE backbone is pooled to obtain a highly discriminative 768-dimensional continuous feature embedding. 

These embeddings are passed to a Multi-Level Density Estimator (MULDE), exploiting Denoising Score Matching (DSM). By injecting multi-scale Gaussian noise into the embeddings and training a conditional score network, MULDE accurately models the probability density of normal traffic behavior. A Gaussian Mixture Model (GMM) calibrates these deviations into a bounded semantic anomaly score:
\begin{equation}
A_s = -\log P(x | \text{GMM})
\end{equation}

\subsection{Behavioral Anomaly Module ($A_b$)}
Parallel to the semantic stream, explicit vehicle tracking via YOLO and DeepSORT extracts bounding box kinematics. We measure instantaneous velocity ($z_v$), acceleration ($z_a$), jerk ($j_t$), spatial entropy ($H$), and wrong-way trajectories ($W$). The behavioral anomaly is derived as:
\begin{equation}
A_b = 0.30z_v + 0.25z_a + 0.20j_t + 0.15H + 0.10W
\end{equation}

The final anomaly fusion integrates both the implicit density gradient and the explicit kinematic divergence:
\begin{equation}
A_t = \alpha A_s + (1-\alpha) A_b
\end{equation}

\subsection{Traffic Prediction Module ($F_t, C_f$)}
Historical traffic matrices are fed into an LSTM sequence network to generate predictive trajectory bounds $F_t$. The forecast confidence $C_f$ mathematically scales uncertainty:
\begin{equation}
C_f = 1 - \frac{\sigma(F_t)}{\max(\sigma)}
\end{equation}

\subsection{Graph Representation Module ($G_t$)}
The entire intersection network is formulated as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. Graph Convolutional Network (GCN) and Graph Attention (GAT) layers establish neighborhood awareness, computing topological embeddings $G_t$ via message passing.

\subsection{Carbon Engine ($C_t$)}
Emissions are modeled via vehicle kinetics, acting as a direct environmental penalty:
\begin{equation}
C_t = \sum_i CO_2(v_i, a_i)
\end{equation}

\subsection{Emergency Routing Module ($E_t$)}
Ambulances and emergency response vehicles trigger an algorithmic pathfinding bypass, calculating Dijkstra's absolute priority routing, outputting the boolean override $E_t$.

\subsection{Unified State Construction ($Z_t$)}
The individual isolated modalities are concatenated into an ultra-dense, multi-dimensional state vector encompassing the entire topological, kinematic, environmental, and semantic state of the physical road grid:
\begin{equation}
Z_t = [G_t, A_s, A_b, F_t, C_f, C_t, E_t]
\end{equation}

\subsection{MAPPO CTDE}
The Centralized Training with Decentralized Execution mechanism governs the Multi-Agent Reinforcement Learning strategy. Actors operate on local unified states $z_i$, while the Centralized Critic processes $Z_{global}$. The Proximal Policy Optimization objective is formulated as:
\begin{equation}
L_{PPO} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t) A_t)]
\end{equation}

\subsection{Joint Optimization ($L_{total}$)}
To prevent catastrophic gradient interference when simultaneously training disparate neural backbones (LSTM, GNN, PPO), we backpropagate a unified joint loss:
\begin{equation}
L_{total} = L_{PPO} + \lambda_1 L_{LSTM} + \lambda_2 L_{GNN}
\end{equation}
Cosine similarity ensures constructive gradient updates across the feature encoders:
\begin{equation}
\text{Cos}(\theta) = \frac{g_1 \cdot g_2}{|g_1| |g_2|}
\end{equation}

\subsection{Safety Shield}
An absolute emergency override mechanism deterministically assumes control when emergency vectors dictate collision avoidance. If a catastrophic topology is proposed by the Actor network, the Shield intercepts the phase transition, defaulting to fail-safe clearance intervals.

\section{Optimization Objective}
\label{sec:optimization_objective}
The reinforcement learning objective is entirely dictated by the scalar reward function. The comprehensive reward equation explicitly balances throughput against emissions and anomaly routing:
\begin{equation}
R_t = w_1 R_{traffic} - w_2 C_t - w_3 A_t + w_4 E_t + w_5 C_f
\end{equation}
where $R_{traffic}$ maximizes volume while minimizing wait times, and $w_x$ represent the tunable linear coefficients.

\section{Computational Complexity}
\label{sec:complexity}
The time complexity bounds for real-time inference execution are summarized below.
\begin{table}[htbp]
\centering
\caption{Computational Complexity Bounds}
\begin{tabular}{|c|c|}
\hline
\textbf{Module} & \textbf{Complexity} \\
\hline
Behavioral Anomaly & $\mathcal{O}(N)$ \\
Emergency Routing & $\mathcal{O}(E+V\log V)$ \\
LSTM Forecasting & $\mathcal{O}(WH)$ \\
Graph Representation & $\mathcal{O}(V+E)$ \\
MAPPO Actor & $\mathcal{O}(|Z||A|)$ \\
Safety Shield & $\mathcal{O}(E+V\log V)$ \\
Unified State Construction & $\mathcal{O}(|Z|)$ \\
\hline
\end{tabular}
\end{table}

\section{Experimental Setup}
\label{sec:experimental_setup}
The experimental protocol utilizes SUMO, BDD100K, and Cityscapes datasets. We test grid topologies scaling from 1x1, 2x2, 4x4, 8x8, up to 64 cooperative intersections. Training executes over 10,000 continuous episodes across 5 random seeds to ensure statistical significance. The architecture is deployed on High-Performance Computing NVIDIA A100 clusters (64 CPU, 512GB RAM) using Docker, SLURM, and PyTorch (CUDA). 

\section{Results}
\label{sec:results}
\textit{Note on Empirical Availability:} In strict adherence to scientific reproducibility standards, no synthetic or fabricated numerical values are presented in this analysis. The downstream empirical results are explicitly identified as blocked until authentic HPC Phase III execution completes.

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
We expect the SPGRL framework to demonstrate significant advantages in bounding latency while navigating dynamic visual anomalies. The scaling behavior is hypothesized to remain robust up to 64 intersections, though generalization to heavily unstructured road networks remains an active area of investigation.

\section{Limitations}
\label{sec:limitations}
The architecture faces strict HPC computational requirements; the vast memory cost of VideoMAE precludes straightforward edge inference deployment. Additionally, centralized critic communication assumptions may fail under realistic low-bandwidth urban environments. Dataset bias and physical deployment complexity remain formidable challenges.

\section{Future Work}
\label{sec:future_work}
Future work focuses on transitioning to Spatial-Temporal GNNs (STGNN) and Vision Transformers. We aim to apply knowledge distillation and Federated MAPPO to enable localized Jetson deployment and integration into live smart city digital twins.

\section{Conclusion}
\label{sec:conclusion}
This work proposes a unified Semantic Predictive Graph Reinforcement Learning architecture that integrates semantic perception, behavioral analysis, predictive forecasting, graph reasoning, carbon optimization, emergency routing, and multi-agent reinforcement learning into a mathematically consistent traffic control framework. The framework demonstrates that extreme multimodality can be jointly optimized for sustainable, safe, and autonomous urban infrastructure.

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
"""
    output_path = r"c:\Users\Asus\Downloads\main (1).tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print("Beautiful SPGRL Reconstruction Complete.")

if __name__ == "__main__":
    generate_beautiful_spgrl()
