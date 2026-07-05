import torch
import torch.nn as nn
import torch.nn.functional as F
from v2.rl.mappo.graph_encoder import build_graph_encoder

class CentralizedCritic(nn.Module):
    """
    Centralized Critic for MAPPO using CTDE.
    Inputs:
        - States: [B, N, 168]
        - Adjacency: [B, N, N]
        - Global Metrics: [B, 5] (Avg Queue, Avg Delay, Avg Carbon, Emergencies, Congestion)
    Outputs:
        - Value: [B, 1] Joint Advantage
    """
    def __init__(self, state_dim: int = 168, encoder_type: str = "gcn", hidden_dim: int = 256):
        super().__init__()
        
        # Pluggable Graph Spatial Encoder
        self.graph_encoder = build_graph_encoder(encoder_type, state_dim, hidden_dim)
        
        # We pool the N nodes into a single global graph embedding [B, hidden_dim]
        # Then concatenate with the 5 global metrics
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, states: torch.Tensor, adj: torch.Tensor, global_metrics: torch.Tensor) -> torch.Tensor:
        """
        states: [B, N, 168]
        adj: [B, N, N]
        global_metrics: [B, 5]
        """
        # 1. Spatially encode the N local states using the Adjacency topology
        # output: [B, N, hidden_dim]
        node_embeddings = self.graph_encoder(states, adj)
        
        # 2. Global Pooling (Mean across N agents) -> [B, hidden_dim]
        graph_embedding = node_embeddings.mean(dim=1)
        
        # 3. Inject explicit Global Metrics
        # [B, hidden_dim + 5]
        combined = torch.cat([graph_embedding, global_metrics], dim=-1)
        
        # 4. Joint Value estimation
        value = self.global_mlp(combined)
        
        return value
