import os
from pathlib import Path

def generate_section4():
    sec4_tex = r"""
\section{Mathematical Formulations}
\label{sec:mathematical_formulations}

\subsection{Semantic Anomaly Formulation}
The semantic anomaly engine operates strictly on unconstrained visual data to capture macroscopic scene volatility. We define the raw input space as continuous video frames $I_t$. Rather than utilizing bounding box heuristics, we employ an offline Video Masked Autoencoder (VideoMAE) to extract a highly discriminative latent embedding $x_s$:
\begin{equation}
x_s = \text{VideoMAE}(I_t)
\end{equation}
where $x_s \in \mathbb{R}^{768}$ captures the dense spatio-temporal kinematics of the intersection. To quantify the deviation of this embedding from normative traffic patterns, we employ a Multi-Level Density Estimator (MULDE):
\begin{equation}
\text{MULDE}(x_s)
\end{equation}
This density is calibrated into a continuous probability distribution utilizing a Gaussian Mixture Model (GMM). The likelihood $p(x_s)$ is defined as:
\begin{equation}
p(x_s) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x_s \mid \mu_k, \Sigma_k)
\end{equation}
where $K$ represents the number of mixture components, $\pi_k$ denotes the mixing coefficient, and $\mathcal{N}$ is the multivariate normal distribution parameterized by mean $\mu_k$ and covariance $\Sigma_k$. The final semantic anomaly score $A_s$ is extracted as the negative log-likelihood of the embedding:
\begin{equation}
A_s = -\log p(x_s)
\end{equation}
A higher $A_s$ value explicitly correlates to severe, out-of-distribution physical anomalies, providing the downstream policy with a high-confidence probabilistic severity measure.

\subsection{Behavioral Anomaly Formulation}
Complementing the implicit semantic analysis, the behavioral anomaly engine explicitly tracks individual vehicular micro-kinematics. Using YOLO object detection coupled with DeepSORT filtering, we extract the precise trajectory state of the $i$-th vehicle over a temporal horizon:
\begin{equation}
s_i = [v_i, a_i, j_i, H_i, W_i]
\end{equation}
where $v_i$ is instantaneous velocity, $a_i$ is acceleration, $j_i$ represents jerk (derivative of acceleration), $H_i$ denotes the spatial trajectory entropy, and $W_i$ is a boolean wrong-way trajectory indicator. 

To fuse these continuous kinematics, we apply z-score normalization, bounding the statistical deviations relative to historical intersection flow baselines. The aggregate behavioral anomaly $A_b$ over the intersection is computed as a weighted linear combination of these normalized divergences:
\begin{equation}
A_b = \alpha_1 z_v + \alpha_2 z_a + \alpha_3 z_j + \alpha_4 H + \alpha_5 W
\end{equation}
where $z_v, z_a,$ and $z_j$ represent the normalized deviations in speed, acceleration, and jerk respectively, while the coefficients $\alpha_x$ govern the relative severity weighting of each kinematic divergence.

\subsection{Traffic Prediction Formulation}
To transition the system from reactive lagging control to proactive planning, short-term temporal forecasting is integrated. A historical sequence matrix is ingested by a Long Short-Term Memory (LSTM) sequence network. The recurrent hidden state update is defined as:
\begin{equation}
h_t = f(x_t, h_{t-1})
\end{equation}
where $h_t$ is the current hidden state encapsulating the temporal dependencies. The future prediction horizon $F_t$ bounding the forecasted volume and queue states is mapped via:
\begin{equation}
F_t = g(h_t)
\end{equation}
Because neural forecasting is inherently stochastic during severe anomalies, we explicitly formulate a prediction confidence score $C_f$ based on the predictive variance $\sigma^2(F_t)$ leveraging Monte Carlo bounds:
\begin{equation}
C_f = 1 - \sigma^2(F_t)
\end{equation}
This rigorous uncertainty estimation dictates whether the downstream reinforcement learning policy should heavily exploit the temporal horizon or revert entirely to instantaneous observation.

\subsection{Graph Representation Formulation}
The physical urban network is topologically structured as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ defines intersections (nodes) and $\mathcal{E}$ defines connecting arterial roads (edges). For each node, we extract a localized feature tensor $X_i$. 

The hidden state of each intersection is recursively updated via Graph Convolutional Networks (GCN) to explicitly pass shockwave matrices between neighbors:
\begin{equation}
h_i^{(l+1)} = \sigma\left( W_s h_i + W_n \sum_{j \in \mathcal{N}_i} h_j \right)
\end{equation}
where $W_s$ represents the self-weight transformation, and $W_n$ represents the neighborhood transformation aggregating adjacent node features. Through subsequent graph attention formulations, asymmetric node weights are assigned based on directional flow severities, producing the final aggregated spatial graph embedding $G_t$. This computational message passing scales efficiently at $\mathcal{O}(|V|+|E|)$.

\subsection{Carbon Emission Formulation}
Optimizing strictly for vehicle throughput fundamentally ignores the massive environmental externalities of intersection idling. Leveraging real-time traffic telemetry, the Carbon Engine translates physical kinematics into continuous emission penalties. Total instantaneous intersection emissions $C_t$ are modeled by aggregating the carbon footprint of all vehicles:
\begin{equation}
C_t = \sum_{i} c_i
\end{equation}
where the individual emission rate $c_i$ is mapped as a non-linear function of speed and acceleration dynamics:
\begin{equation}
c_i = f(v_i, a_i)
\end{equation}
This explicitly penalizes violent start-stop shockwaves and severe stationary idling. The carbon formulation is integrated as a continuous penalty $R_c$ within the terminal reward function.

\subsection{Emergency Routing Formulation}
Emergency preemption supersedes all stochastic optimization. Upon an emergency request, the routing algorithm evaluates the arterial sub-graph. The optimal priority routing path $P^*$ minimizes total traversal cost:
\begin{equation}
P^* = \text{argmin}_{\text{cost}(P)}
\end{equation}
The pathfinding protocol operates completely outside the neural network loop via Dijkstra's algorithm, maintaining a strict deterministic complexity of $\mathcal{O}(|E| + |V| \log |V|)$. The module outputs an emergency priority indicator $E_t$, signaling the presence of an active preemption requirement on the local node.

\subsection{Unified State Construction}
The individual observation modalities are fused into the exact SPGRL Unified State vector $Z_t$:
\begin{equation}
Z_t = \left[ G_t, A_s, A_b, F_t, C_f, C_t, E_t \right]
\end{equation}
This fusion is mathematically possible due to the uniform continuous bounding of the disparate multi-modal vectors. The state construction prevents catastrophic dimension mismatch by applying explicit normalization across the highly dense concatenated array. By binding semantic perception directly to topological graph embeddings and predictive confidence, the high-dimensionality observation space is rigorously synchronized, presenting the optimal Markov state to the decision policy.

\subsection{Multi-Agent PPO Formulation}
The control layer utilizes Multi-Agent Proximal Policy Optimization (MAPPO), structured around the Centralized Training with Decentralized Execution (CTDE) paradigm. The decentralized actor policy $\pi_\theta(a \mid s)$ outputs the optimal signal phase transition probability given localized states, while the centralized critic estimates the value function $V_\phi(s)$ leveraging the global joint state vector. 

The temporal difference error yields the advantage estimator $A_t$. The PPO clipping objective ensures monotonic trust-region improvements, bounding the policy update via:
\begin{equation}
L_{PPO} = \mathbb{E} \left[ \min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right) \right]
\end{equation}
CTDE inherently resolves the non-stationarity issues traditionally plaguing isolated single-agent RL traffic controllers.

\subsection{Joint Optimization Formulation}
Training the independent GNN, LSTM, and PPO backbones sequentially risks severe distributional shift. The SPGRL framework executes joint optimization to backpropagate unified gradients:
\begin{equation}
L_{total} = L_{PPO} + \lambda_1 L_{LSTM} + \lambda_2 L_{GNN} + \lambda_3 L_{anomaly}
\end{equation}
To explicitly prevent gradient conflict mitigation—where PPO actor optimization degrades the upstream feature extraction topologies—we monitor gradient similarity using cosine proximity:
\begin{equation}
\cos(\theta) = \frac{g_1 \cdot g_2}{|g_1| |g_2|}
\end{equation}
This prevents catastrophic interference, ensuring that updates prioritizing throughput do not systematically destroy the semantic severity mappings.

\subsection{Safety Shield Formulation}
To guarantee fail-safe physical operation, the final action actuation is strictly gated by a deterministic rule-based Safety Shield. If an emergency priority indicator is active ($E_t = 1$), the framework mandates deterministic intervention:
\begin{equation}
\pi(a \mid s) \rightarrow \text{override} \rightarrow \text{safe action}
\end{equation}
The overridden optimal safe action $a^*$ is mathematically defined to maximize the collision avoidance boundaries and enforce green-wave generation:
\begin{equation}
a^* = \text{argmax safety}(a)
\end{equation}
This deterministic control completely bypasses neural hallucination, ensuring absolute physical safety compliance.

\subsection{Computational Complexity Analysis}
The bounded complexity of the SPGRL components governs their suitability for real-time edge execution. The asymptotic time complexities are documented in Table~\ref{tab:complexity_spgrl}.

\begin{table}[htbp]
\centering
\caption{Module Complexity Analysis}
\label{tab:complexity_spgrl}
\begin{tabular}{|l|l|}
\hline
\textbf{Module} & \textbf{Complexity} \\
\hline
Behavioral & $\mathcal{O}(N)$ \\
Semantic & $\mathcal{O}(KD)$ \\
LSTM & $\mathcal{O}(WH)$ \\
GNN & $\mathcal{O}(V+E)$ \\
MAPPO & $\mathcal{O}(|Z||A|)$ \\
Emergency Routing & $\mathcal{O}(E+V\log V)$ \\
Unified State & $\mathcal{O}(|Z|)$ \\
\hline
\end{tabular}
\end{table}

The runtime complexity remains strictly linear with respect to bounding box extractions ($\mathcal{O}(N)$) and log-linear for emergency priority pathfinding. Memory complexity is heavily dominated by the sequence parameters in the LSTM ($\mathcal{O}(WH)$) and the high-dimensional spatial topology parameters in the GNN ($\mathcal{O}(V+E)$). However, the highly decoupled acyclic dependency graph ensures optimal scalability and profound HPC deployment feasibility on distributed GPU accelerators.
"""
    
    project_root = Path(__file__).resolve().parents[0]
    path = project_root / "SPGRL_IEEE_FINAL_MANUSCRIPT.tex"
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Remove \end{document}
    content = content.replace(r"\end{document}", "")
    
    # Append the new section
    content += sec4_tex
    content += "\n\\end{document}\n"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    generate_section4()
