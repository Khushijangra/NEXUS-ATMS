import torch
import torch.nn as nn
from typing import Dict, List

class MAPPOCommunication(nn.Module):
    """
    Handles message passing between agents. Extracts critical features from 
    the 168D state, including forecasts and confidences, and distributes 
    them along the adjacency matrix.
    """
    def __init__(self, msg_dim: int = 16):
        super().__init__()
        # We extract 6 features per node (Queue, Delay, Carbon, Emergency, Forecast, Confidence)
        self.extract_dim = 6
        
        # Message encoding
        self.msg_encoder = nn.Linear(self.extract_dim, msg_dim)
        
    def extract_features(self, states: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 6 key communication features from the 168D state.
        state layout reference:
        [0:1] Queue
        [1:2] Delay
        [2:52] Camera (50)
        [52:102] Graph (50)
        [102:166] LSTM Forecast (64)
        [166:167] Emergency
        [167:168] Confidence
        
        Note: Carbon is derived from Queue. For pure state extraction:
        Queue (0), Delay (1), Forecast (mean of 102:166), Emergency (166), Confidence (167).
        """
        batch_size, num_agents, _ = states.shape
        features = torch.zeros((batch_size, num_agents, self.extract_dim), device=states.device)
        
        # Queue
        features[:, :, 0] = states[:, :, 0]
        # Delay
        features[:, :, 1] = states[:, :, 1]
        # Carbon (proxy derived from queue)
        features[:, :, 2] = states[:, :, 0] * 0.05
        # Emergency
        features[:, :, 3] = states[:, :, 166]
        # Forecast summary (using mean of 64D embedding as proxy for severity)
        features[:, :, 4] = states[:, :, 102:166].mean(dim=-1)
        # Confidence
        features[:, :, 5] = states[:, :, 167]
        
        return features
        
    def forward(self, states: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        states: [B, N, 168]
        adj: [B, N, N]
        Returns aggregated messages [B, N, msg_dim]
        """
        # 1. Extract [B, N, 6]
        raw_features = self.extract_features(states)
        
        # 2. Encode to [B, N, msg_dim]
        encoded_msgs = torch.relu(self.msg_encoder(raw_features))
        
        # 3. Message Passing (sum of neighbors)
        # norm degree
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        norm_adj = adj / degree
        
        # [B, N, N] @ [B, N, msg_dim] -> [B, N, msg_dim]
        aggr_msgs = torch.bmm(norm_adj, encoded_msgs)
        
        return aggr_msgs
