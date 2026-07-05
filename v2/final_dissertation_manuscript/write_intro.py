import os
from pathlib import Path

def generate_introduction():
    intro_tex = r"""
\section{Introduction}

\subsection{Urban Traffic Signal Control Problem}
Unprecedented urbanization has elevated traffic congestion into one of the most critical inefficiencies within modern smart city infrastructure. Global estimates indicate that intersection gridlock induces severe economic and environmental penalties, heavily escalating wasted fuel, diminishing logistical productivity, and drastically elevating vehicular carbon emissions. Beyond economic attrition, volatile intersections present acute safety hazards, generating compounding delays for emergency response vehicles and acting as dense catalysts for vehicular collisions. Addressing this structural volatility is paramount for establishing sustainable, secure, and highly efficient urban environments.

\subsection{Existing Traffic Signal Control Methods}
Conventional intelligent transportation systems heavily rely on legacy deterministic architectures, primarily fixed-time controllers, actuated sensors, and macroscopic adaptive systems such as SCOOT and SCATS. These systems fundamentally operate on reactive heuristics—modifying cyclic phase lengths strictly based on immediate upstream induction loop occupancy. Consequently, these architectures are structurally incapable of adapting to macro-level stochastic perturbations, unpredictable asymmetric load distributions, or complex spatial topological shifts, rendering them highly rigid in dynamic metropolitan environments.

\subsection{Reinforcement Learning Traffic Signal Control}
To address the inflexibility of deterministic control, the field has extensively transitioned toward deep reinforcement learning paradigms. Architectures leveraging Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and Advantage Actor-Critic (A2C) have demonstrated significant efficacy in continuous state-action mapping. However, standard Multi-Agent Reinforcement Learning (MARL) paradigms, including baseline MAPPO, traditionally rely exclusively on unimodal numerical state matrices, such as queue lengths and waiting times. This causes catastrophic failure when unquantified physical events—such as collisions or erratic pedestrian crossings—disrupt the numerical flow parameters, blinding the agent to the actual visual ground truth.

\subsection{Existing Anomaly Detection Systems}
Parallel efforts to mitigate intersection hazards rely on isolated anomaly detection systems. Early detector-based mechanisms utilizing radar lack semantic framing, while standard computer vision architectures (like YOLO) struggle with dense, unconstrained overlapping occlusions. Conversely, classical trajectory-based algorithms and semantic reconstruction autoencoders suffer heavily from identity mapping failures, requiring massive labeled datasets and failing to produce a calibrated, probabilistic severity gradient that a downstream controller can actively interpret.

\subsection{Existing Prediction Methods}
Proactive routing requires accurately mapping future traffic states, a domain dominated by ARIMA models, Long Short-Term Memory (LSTM) sequence networks, and Spatio-Temporal Graph Convolutional Networks (STGCN). While highly capable of offline forecasting, sequence modeling is rarely integrated tightly into the instantaneous, real-time observation manifold of a decentralized RL policy, isolating the temporal horizon from the actual phase optimization process.

\subsection{Existing Graph-based Traffic Control}
Urban networks are fundamentally topological, prompting the development of graph-based control paradigms such as Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), CoLight, and PressLight. These mechanisms effectively resolve spatial coordination by passing hidden states between neighboring intersections. However, graph networks alone cannot resolve localized visual anomalies or route priority vehicles without manual heuristic overrides, remaining vulnerable to intra-intersection physical disruptions.

\subsection{Existing Sustainable Traffic Control}
The environmental toll of idling vehicles has spurred research into eco-routing and fuel optimization. Yet, continuous carbon-aware traffic control remains heavily under-explored. In most deployments, emission minimization is relegated to post-hoc analysis rather than actively penalized as a continuous mathematical component of the instantaneous reinforcement learning reward function.

\subsection{Existing Emergency Vehicle Priority Systems}
Modern intersections require strict, deterministic routing for ambulances and fire engines. Existing systems rely on dedicated roadside hardware, RFID, or V2X communication to trigger phase preemption. However, these heuristic routing systems lack native integration with visual anomaly engines and entirely disrupt the RL policy's established state-action manifold, causing massive secondary congestion shockwaves once the emergency vehicle clears the junction.

\subsection{Research Gap}
An extensive review of the literature explicitly establishes that current urban infrastructure models lack semantic perception, behavioral perception, sequence prediction, graph reasoning, carbon optimization, and emergency routing within a single integrated framework. The absence of this unified optimization forces modern cities to operate disjointed subsystems. No existing architecture achieves joint learning of deep vision and topological routing while providing deterministic safety shielding against neural hallucinations.

\subsection{Proposed SPGRL Framework}
To bridge this critical translational gap, this paper proposes the Semantic Predictive Graph Reinforcement Learning (SPGRL) framework. The architectural pipeline initiates with the ingestion of Raw Traffic Video, which is processed offline through a Video Masked Autoencoder (VideoMAE) to yield 768-dimensional kinematic features. These features are mapped via Multi-Level Density Estimation (MULDE) and a Gaussian Mixture Model (GMM) to extract a continuous Semantic Anomaly metric ($A_s$). Simultaneously, the video stream undergoes explicit YOLO and DeepSORT tracking to capture bounding box kinematics, generating a continuous Behavioral Anomaly metric ($A_b$).

Concurrently, historical numerical traffic matrices are processed by an LSTM sequence model to extract predictive trajectory bounds ($F_t$) and confidence weights ($C_f$). The topological Neighbor Graph is resolved via a GNN into spatial embeddings ($G_t$). An integrated Carbon Engine dynamically calculates emission penalties ($C_t$), while an Emergency Routing algorithm calculates absolute pathfinding priority ($E_t$).

These multi-modal streams are natively concatenated into a highly dense Unified State ($Z_t$). This unified manifold is fed directly into a Multi-Agent PPO architecture, protected at the execution level by a deterministic Safety Shield to dictate final Signal Control without catastrophic interference.

\subsection{Contributions}
The primary technical contributions of this research are formulated as follows:
\begin{enumerate}
    \item We propose the first Semantic Predictive Graph Reinforcement Learning (SPGRL) framework, fully synthesizing unconstrained vision, predictive modeling, and topological routing for urban traffic control.
    \item We introduce a dual-stream anomaly architecture that mathematically fuses implicit semantic video perception with explicit behavioral trajectory analysis.
    \item We develop a mathematically rigorous VideoMAE-MULDE-GMM semantic anomaly pipeline to extract density-based scene volatility without frame-level annotation.
    \item We formulate a complementary behavioral anomaly engine utilizing YOLO and DeepSORT to quantify micro-kinematic divergence (e.g., erratic acceleration, wrong-way driving).
    \item We integrate LSTM sequence forecasting directly into the reinforcement learning state space, enabling proactive phase transitioning.
    \item We develop a graph-based MAPPO coordination mechanism leveraging Centralized Training with Decentralized Execution (CTDE) to resolve multi-intersection topology.
    \item We propose a continuous carbon-aware optimization strategy embedded natively within the multi-agent reward function.
    \item We introduce a deterministic emergency Safety Shield to preempt stochastic network exploration and guarantee absolute collision avoidance.
    \item We formulate a unified multimodal state representation ($Z_t$) capable of joint optimization without inducing catastrophic gradient interference across disparate neural backbones.
\end{enumerate}

\subsection{Paper Organization}
The remainder of this manuscript is organized as follows. Section II details the proposed SPGRL framework, explicitly mapping the layered architecture and module dependencies. Section III formulates the rigorous mathematical foundations underpinning the semantic, behavioral, predictive, and graph components. Section IV outlines the experimental setup and evaluation protocol. Section V presents the anticipated empirical results and analysis. Section VI discusses overarching systemic implications, while Section VII details architectural limitations. Finally, Section VIII concludes the paper.
"""
    
    project_root = Path(__file__).resolve().parents[0]
    path = project_root / "SPGRL_IEEE_FINAL_MANUSCRIPT.tex"
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Remove \end{document}
    content = content.replace(r"\end{document}", "")
    
    # Append the new section
    content += intro_tex
    content += "\n\\end{document}\n"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    generate_introduction()
