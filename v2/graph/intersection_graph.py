import numpy as np
class IntersectionGraph:
    def __init__(self, adj):
        self.adj = adj
        
    def compute_degree_matrix(self):
        return np.diag(np.sum(self.adj, axis=1))
        
    def compute_laplacian(self):
        D = self.compute_degree_matrix()
        return D - self.adj
        
    def compute_shortest_paths(self):
        from scipy.sparse.csgraph import floyd_warshall
        return floyd_warshall(self.adj, directed=True)
        
    def compute_centrality(self):
        deg = np.sum(self.adj, axis=1)
        return deg / (len(self.adj) - 1)
