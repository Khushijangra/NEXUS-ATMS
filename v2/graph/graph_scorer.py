import torch
import torch.nn as nn
import networkx as nx
import numpy as np

class GraphScorer(nn.Module):
    """
    Computes a 64-D graph embedding (Gt) representing the road network state.
    Currently maps NetworkX centrality metrics into a 64-dimensional feature space.
    """
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        # Map 4 centrality metrics (degree, betweenness, clustering, closeness) to 64D
        self.projection = nn.Linear(4, embed_dim)
        
        # Build the dummy road network (1 intersection, 4 arms)
        self.G = nx.star_graph(4)
        
    def forward(self) -> torch.Tensor:
        # Extract features from NetworkX
        deg = nx.degree_centrality(self.G)
        betw = nx.betweenness_centrality(self.G)
        clust = nx.clustering(self.G)
        close = nx.closeness_centrality(self.G)
        
        # Aggregate features for the central intersection node (node 0)
        feats = [deg[0], betw[0], clust[0], close[0]]
        x = torch.tensor([feats], dtype=torch.float32)
        
        # Project to 64-D embedding
        gt = self.projection(x)
        return gt
