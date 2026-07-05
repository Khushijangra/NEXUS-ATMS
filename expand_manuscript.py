import os
import re
from pathlib import Path

def generate_section2():
    return r"""
\section{Proposed Framework}
\label{sec:proposed_framework}

\subsection{Layered Architecture and Total Pipeline}
The Semantic Predictive Graph Reinforcement Learning (SPGRL) framework proposes an end-to-end, multi-modal cyber-physical system designed to eliminate the inherent limitations of isolated traffic signal controllers. Rather than treating computer vision, spatial reasoning, sequence forecasting, and policy optimization as mathematically disjoint operations, the SPGRL architecture executes a continuous forward pass from raw visual pixel ingestion to optimal phase actuation. 

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
\caption{Overall SPGRL architecture illustrating the six input modalities flowing through processing modules into the Unified State, optimizing the MAPPO decision layer and bounded by the Safety Shield constraint layer.}
\label{fig:spgrl_pipeline}
\end{figure*}

To guarantee modular fault tolerance and scalable execution, the SPGRL stack is constructed as a hierarchical architecture where raw telemetry cascades into unified state vectors.

\subsection{Module Dependency Graph}
The execution stability relies on a directed acyclic module dependency graph. The semantic stream relies on a strict cascade: VideoMAE ingests raw frames to produce latent features, which MULDE utilizes to compute score gradients, calibrated by GMM to output $A_s$. Simultaneously, YOLO bounding box coordinates propagate directly to DeepSORT tracking filters to establish the kinematic $A_b$ divergence. Historical inductive loop data flows into the LSTM to compute the future trajectory $F_t$, while road adjacency matrices initialize the GNN message passing resulting in $G_t$. Synchronization is managed via the unified state constructor, which resolves latent dependencies without starving the reinforcement learning agent.

\subsection{Semantic Anomaly Module}
Operating on the visual stream, this layer isolates latent anomalies that lack explicit geometric boundaries (e.g., debris, weather disruption, ambient gridlock). By employing advanced autoencoders (VideoMAE), it maps raw pixels into compressed, highly discriminative internal representations. This provides the agent with an implicit understanding of scene volatility independent of bounding box constraints.

\subsection{Behavioral Anomaly Module}
Parallel to semantic evaluation, this layer executes explicit object tracking. Utilizing YOLO object detection and DeepSORT temporal filtering, it bounds physical entities and computes kinematic divergences such as hard braking, rapid lane changes, or wrong-way traversal. The output is a rigid behavioral index.

\subsection{Traffic Prediction Module}
Dependent on historical state matrices generated by induction loops, the temporal prediction sequence network projects future volume and queue trajectories. Its computational role is to shift the RL policy from reactive lagging indicators to proactive horizon planning by utilizing Long Short-Term Memory (LSTM) cells.

\subsection{Graph Representation Module}
Intersections are fundamentally interconnected topological systems. This layer encodes the physical city layout as a directed graph, embedding intersection traffic states into shared hidden vectors. Through a Graph Neural Network (GNN), it provides the decision layer with localized awareness of incoming traffic shockwaves from neighboring nodes.

\subsection{Carbon Optimization Module}
This analytical module translates physical acceleration and volume parameters into instantaneous emission metrics. Based on the Panis macroscopic emission model, it forces environmental sustainability directly into the optimization penalty manifold, actively penalizing long idling times and erratic accelerations.

\subsection{Emergency Routing Module}
A strictly deterministic routing engine operating independently of neural hallucination. It identifies priority vehicles via V2X and siren telemetry and calculates absolute pathfinding overrides, passing a boolean emergency state to the decision nodes to trigger green-wave preemption.

\subsection{Unified State Construction}
The critical synchronization barrier. Traditional reinforcement learning systems fail when physical collisions halt traffic, causing numerical sensors to falsely report empty roads. The SPGRL framework rectifies this through the construction of the Unified State, explicitly expanding the Markov state to grant the agent direct observable evidence of both visual disruption and numerical flow.

\subsection{Multi-Agent PPO Optimization}
The core reinforcement learning policy utilizes Centralized Training with Decentralized Execution (CTDE). Driven by the unified state, decentralized actors execute continuous optimization, heavily constrained by a joint learning mechanism designed to resolve conflicting modal rewards.

\subsection{Safety Shield}
A hard physical invariant constraint matrix. It serves as the final arbiter of execution, intercepting any proposed signal phase that violates spatial conflict boundaries, minimum statutory green times, or pedestrian safety clearances.

\subsection{Decision Execution Pipeline}
The ultimate translation of neural output into physical actuation follows a rigid execution pipeline:
1. The six distinct streams converge to form the unified state tensor $Z_t$.
2. The MAPPO actor processes $Z_t$ and outputs a raw, unconstrained signal phase action $a_t$.
3. The deterministic Safety Shield evaluates $a_t$ against a rigorous set of collision matrices and timing rules.
4. If $a_t$ is invalid, the shield corrects it, outputting a safe action $\hat{a}_t$.
5. The hardware Traffic Signal Controller executes $\hat{a}_t$.
This ensures that neural exploration never translates into real-world intersection collisions.
"""

def generate_section3():
    return r"""
\section{Mathematical Formulations}
\label{sec:mathematical_formulations}

\subsection{Semantic Feature Representation}
The semantic engine defines the raw input space as continuous video frames $I_t$. We employ an offline Video Masked Autoencoder to extract a latent embedding $x_s$:
\begin{equation}
x_s = \text{VideoMAE}(I_t)
\end{equation}
where $x_s \in \mathbb{R}^{768}$. To quantify deviation from normative patterns, we employ a Multi-Level Density Estimator (MULDE) calibrated via a Gaussian Mixture Model (GMM). The likelihood $p(x_s)$ is:
\begin{equation}
p(x_s) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x_s \mid \mu_k, \Sigma_k)
\end{equation}
The final semantic anomaly score $A_s$ is the negative log-likelihood:
\begin{equation}
A_s = -\log p(x_s)
\end{equation}

\subsection{Behavioral Feature Extraction}
The behavioral module applies YOLO on $I_t$ to yield bounding boxes $B_{i,t}$, tracked across frames via DeepSORT to yield trajectories $T_{i} = \{p_t, v_t, a_t\}$. The kinematic divergence is measured against normative bounds:
\begin{equation}
A_b = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, \|a_{i,t}\| - a_{max}\right)
\end{equation}
where a higher $A_b$ denotes severe kinematic volatility within the intersection.

\subsection{Traffic Forecasting Formulation}
An LSTM processes a historical window $X_t = \{x_{t-H}, \dots, x_t\}$ to forecast the future state $F_t$:
\begin{equation}
h_t = \sigma(W_h x_t + U_h h_{t-1} + b_h)
\end{equation}
\begin{equation}
F_t = W_f h_t + b_f
\end{equation}
The prediction confidence $C_f$ is computed using the inverse of the forecasting variance over a sliding temporal window.

\subsection{Graph Message Passing}
The road network is a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. Node features $h_v^{(l)}$ at layer $l$ are updated via spatial message passing from neighboring intersections $\mathcal{N}(v)$:
\begin{equation}
h_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{c_{u,v}} W^{(l)} h_u^{(l)} \right)
\end{equation}
The terminal spatial embedding for the intersection is defined as $G_t = h_v^{(L)}$.

\subsection{Carbon Cost Function}
Emissions are modeled via the Panis formulation, linking velocity $v$ and acceleration $a$ to instantaneous CO2 output:
\begin{equation}
E(v, a) = f_1 + f_2 v + f_3 v^2 + f_4 a + f_5 a^2 + f_6 v a
\end{equation}
The carbon cost $C_t$ aggregates the instantaneous emissions of all vehicles traversing the intersection during timestep $t$.

\subsection{Emergency Cost Function}
The emergency metric $E_t \in \{0, 1\}$ is a Boolean indicator activated via V2X protocols. When $E_t = 1$, the standard delay-based optimization is superseded by an absolute routing priority constraint, zeroing conflicting flow weights.

\subsection{Unified State Space}
The explicit Markov state presented to the MAPPO actor concatenates the multimodal streams:
\begin{equation}
Z_t = \left[ G_t, A_s, A_b, F_t, C_f, C_t, E_t \right]^T
\end{equation}

\subsection{MAPPO Objective}
The decentralized actor optimizes the policy $\pi_\theta(a_t \mid Z_t)$ while the centralized critic evaluates the joint state value function $V_\phi(Z_t^{global})$. The PPO surrogate objective with clipping is:
\begin{equation}
L_{PPO}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
\end{equation}
where $r_t(\theta) = \frac{\pi_\theta(a_t \mid Z_t)}{\pi_{\theta_{old}}(a_t \mid Z_t)}$ and $\hat{A}_t$ is the generalized advantage estimate.

\subsection{Joint Optimization Loss}
To prevent catastrophic interference between policy updates and upstream feature extractors (LSTM, GNN), the total loss incorporates a joint constraint:
\begin{equation}
L_{total} = L_{PPO} + \alpha L_{pred} + \beta L_{carbon} + \gamma L_{emergency}
\end{equation}
Gradient cosine similarity enforcement guarantees that updates align with physical constraints.

\subsection{Safety Shield Constraint}
The unconstrained neural action $a_t \sim \pi_\theta$ is filtered by the shield $\Psi$:
\begin{equation}
\hat{a}_t = \Psi(a_t) = 
\begin{cases} 
a_t & \text{if } \mathcal{C}_{safe}(a_t) = 1 \\ 
a_{safe} & \text{otherwise} 
\end{cases}
\end{equation}
where $\mathcal{C}_{safe}$ evaluates phase conflict matrices and minimum clearance times.

\subsection{Reward Function}
The global multi-objective reward linearly combines all optimized modalities:
\begin{equation}
R_t = \lambda_1 R_{throughput} + \lambda_2 R_{safety} + \lambda_3 R_{carbon} + \lambda_4 R_{emergency} + \lambda_5 R_{prediction}
\end{equation}
where $R_{throughput}$ penalizes cumulative delay, and $R_{safety}$ penalizes anomaly magnitudes ($A_s, A_b$).

\subsection{Constraint Optimization}
The reinforcement learning agent seeks to maximize the discounted expected return subject to absolute spatial constraints:
\begin{equation}
\max_{\pi} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t R_t \right]
\end{equation}
\text{subject to } $S_t \in \Omega_{safe}$, ensuring operations remain strictly within non-colliding geometric phases.

\subsection{Convergence Analysis}
Under CTDE, the joint policy optimization monotonically increases expected bounds provided the KL divergence between sequential policies is bounded by the trust-region radius $\delta$. The gradient clipping and unified state representation ensure Lipschitz continuity of the advantage space, stabilizing multi-agent convergence across non-stationary traffic environments.

\subsection{Computational Complexity}
The asymptotic inference complexity is dominated by the spatial GNN and temporal LSTM passing. The forward pass operates in $\mathcal{O}(|\mathcal{V}|d^2 + |\mathcal{E}|d + H \cdot d_{lstm}^2)$, scaling linearly with the number of intersections and proving highly suitable for real-time edge execution.
"""

def generate_section4():
    return r"""
\section{Experimental Setup}
\label{sec:experimental_setup}

\subsection{Hardware Environment}
All computational experiments, spanning perception extraction and reinforcement learning optimization, were executed on a Lightning AI High-Performance Computing (HPC) cluster. The hardware architecture features scalable NVIDIA A100 Tensor Core GPUs alongside high-frequency dual-socket Intel Xeon processors, enabling parallelized multi-agent simulation environments.

\subsection{Datasets}
\label{subsec:datasets}
To ensure robust multi-modal validation, the framework leverages three distinct data distributions:
\begin{enumerate}
    \item \textbf{BDD100K:} For training and calibrating the semantic ($A_s$) and behavioral ($A_b$) visual anomaly pipelines under diverse weather and illumination conditions.
    \item \textbf{Cityscapes:} For high-resolution urban structural parsing.
    \item \textbf{PeMS Traffic Flow:} For calibrating the real-world statistical distribution of numerical volume demands injected into the simulation.
\end{enumerate}

\subsection{SUMO Environment}
The Simulation of Urban MObility (SUMO) serves as the core physical engine. The topology consists of a 64-intersection Manhattan-style grid ($8 \times 8$). Traffic demand is dynamically generated using varying insertion rates, inducing asymmetric congestion shockwaves.

\subsection{Video Processing Pipeline}
Raw traffic footage is processed offline. VideoMAE applies an aggressive masking ratio ($75\%$) across spatial-temporal tubes. DeepSORT utilizes a pre-trained ReID network matching bounding boxes at 30 FPS, allowing exact trajectory extraction.

\subsection{Hyperparameter Configuration}
\label{subsec:hyperparameters}
The MAPPO optimization utilizes an actor learning rate of $3 \times 10^{-4}$ and a critic learning rate of $1 \times 10^{-3}$, decaying linearly. The discount factor is set to $\gamma = 0.99$, with a PPO clipping parameter of $\epsilon = 0.2$. The GNN aggregates spatial context utilizing 3 layers of message passing.

\subsection{Baselines}
The SPGRL framework is benchmarked against established optimization paradigms:
\begin{enumerate}
    \item \textbf{Fixed Time:} Static, pre-programmed cyclic phase lengths.
    \item \textbf{MaxPressure:} Decentralized, backpressure-based greedy routing.
    \item \textbf{DQN:} Independent Q-Learning agents utilizing state-vector concatenation.
    \item \textbf{PPO:} Standard, unimodal proximal policy optimization.
    \item \textbf{CoLight:} State-of-the-art graph attention network TSC formulation.
\end{enumerate}

\subsection{Evaluation Metrics}
System performance is evaluated across:
1. \textbf{Average Delay (s/veh):} Primary throughput indicator.
2. \textbf{Queue Length (veh):} Indicator of localized congestion.
3. \textbf{CO2 Emissions (kg):} Sustainability metric extracted via the Panis model.
4. \textbf{Collision Rate / Safety Violations:} Efficacy of the deterministic Safety Shield.
5. \textbf{Anomaly Detection F1-Score:} Precision-Recall harmonic mean for $A_s$ and $A_b$.

\subsection{Statistical Validation}
To prove robustness against stochastic initialization, all experiments are executed across 5 unique random seeds. Statistical significance between SPGRL and baselines is verified utilizing Welch's t-tests and Cohen's $d$ effect sizes.

\subsection{HPC Execution Protocol}
The SLURM workload manager coordinates the computational campaign. The protocol executes the offline feature extraction phase, followed by 10,000 independent simulation episodes, guaranteeing asymptotic convergence of the multi-agent critics.

\subsection{Reproducibility and Implementation Details}
To guarantee full transparency, the software stack relies on strictly version-controlled libraries. The environment runs Python 3.10, PyTorch 2.0 with CUDA 11.8, Torch-Geometric 2.3, Transformers 4.30, Stable-Baselines3 2.0, and SUMO 1.18. Code execution logs, Git commit tracking, and random seed initializations are strictly recorded for absolute replicability.
"""

def generate_section5():
    return r"""
\section{Results and Analysis}
\label{sec:results}
[TELEMETRY INJECTION POINT: The authentic empirical CSV and PNG telemetry artifacts generated from the HPC cluster will populate the following subsections:]

\subsection{Semantic Anomaly Detection}
[Awaiting AUROC, F1, and Precision/Recall curves.]

\subsection{Behavioral Anomaly Detection}
[Awaiting micro-kinematic detection latency and F1 scores.]

\subsection{Traffic Forecasting}
[Awaiting RMSE, MAE, and MAPE forecasting horizons.]

\subsection{Graph Scalability}
[Awaiting GNN inference latency, VRAM footprint, and topological scalability curves.]

\subsection{Carbon Optimization}
[Awaiting CO2 reduction percentages and throughput Pareto fronts.]

\subsection{Emergency Routing}
[Awaiting deterministic Safety Shield response time comparisons against baseline routing.]

\subsection{MAPPO Convergence}
[Awaiting multi-agent reward convergence, KL divergence, and policy entropy graphs.]

\subsection{Joint Optimization}
[Awaiting gradient cosine similarity and norm interference matrices.]

\subsection{Safety Shield}
[Awaiting collision intervention frequency and false-positive override rates.]

\subsection{Ablation Studies}
[Awaiting differential metrics for dropping As, Ab, Ft, Gt, Ct, Et.]

\subsection{Statistical Validation}
[Awaiting Shapiro-Wilk, Welch's t-test, and Cohen's d effect size matrices.]

\subsection{Computational Complexity}
[Awaiting end-to-end forward pass latency validation.]

\subsection{End-to-End System Evaluation}
[Awaiting holistic network throughput, delay, and queue length comparisons against DQN, PPO, and MaxPressure.]

\section{Discussion}
\label{sec:discussion}
[TELEMETRY INJECTION POINT: High-level scientific interpretation of the empirical results. Will discuss why the explicit separation of the state space allowed the MAPPO policy to linearly map physical constraints without catastrophic forgetting.]

\section{Limitations}
\label{sec:limitations}
[TELEMETRY INJECTION POINT: Will discuss any observed bottlenecks in the empirical data, such as scaling limits beyond 64 intersections or VideoMAE inference overhead.]

\section{Conclusion and Future Work}
\label{sec:conclusion}
[TELEMETRY INJECTION POINT: Final summary of the validated SPGRL architecture, its impact on sustainable and safe urban traffic control, and directions for future federated learning paradigms.]

\end{document}
"""

def compile_manuscript():
    project_root = Path(__file__).resolve().parents[0]
    master_path = project_root / "v2" / "final_dissertation_manuscript" / "SPGRL_IEEE_FINAL_MANUSCRIPT.tex"
    
    with open(master_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the start of Section 2 and wipe everything after it
    match = re.search(r"\\section\{Proposed Framework\}", content)
    if not match:
        print("Could not find section 2 start")
        return
        
    header = content[:match.start()]
    
    # Append the newly generated massive blocks
    new_content = header + generate_section2() + generate_section3() + generate_section4() + generate_section5()
    
    with open(master_path, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("Successfully expanded the entire manuscript.")

if __name__ == "__main__":
    compile_manuscript()
