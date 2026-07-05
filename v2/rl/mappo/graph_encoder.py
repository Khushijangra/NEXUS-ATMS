import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphEncoder(nn.Module):
    """
    Abstract Base for pluggable Graph Encoders.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, in_features]
        adj: [B, N, N]
        returns: [B, N, out_features]
        """
        raise NotImplementedError

class GCNEncoder(GraphEncoder):
    """
    Standard Graph Convolutional Network (GCN) layer.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # A_hat = A + I
        batch_size, num_nodes, _ = adj.shape
        eye = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        a_hat = adj + eye
        
        # D_hat^-1/2
        degree = a_hat.sum(dim=-1)
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        
        d_mat = torch.diag_embed(d_inv_sqrt)
        
        # Normalized adj: D^-1/2 * A_hat * D^-1/2
        norm_adj = torch.bmm(torch.bmm(d_mat, a_hat), d_mat)
        
        # Graph convolution: norm_adj * (X * W)
        support = self.linear(x)
        out = torch.bmm(norm_adj, support)
        return F.relu(out)

class GraphSAGEEncoder(GraphEncoder):
    """
    GraphSAGE-style inductive encoder (Placeholder for future ablations).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.lin_self = nn.Linear(in_features, out_features)
        self.lin_neigh = nn.Linear(in_features, out_features)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Simple mean aggregation
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        neigh_aggr = torch.bmm(adj, x) / degree
        
        out = self.lin_self(x) + self.lin_neigh(neigh_aggr)
        return F.relu(out)
        
def build_graph_encoder(encoder_type: str, in_features: int, out_features: int) -> GraphEncoder:
    if encoder_type == "gcn":
        return GCNEncoder(in_features, out_features)
    elif encoder_type == "graphsage":
        return GraphSAGEEncoder(in_features, out_features)
    else:
        raise ValueError(f"Unknown graph encoder type: {encoder_type}")
