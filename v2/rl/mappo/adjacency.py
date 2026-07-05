import numpy as np

class AdjacencyBuilder:
    """
    Generates adjacency matrices for different multi-agent traffic topologies.
    """
    @staticmethod
    def build(topology: str, num_nodes: int) -> np.ndarray:
        if topology == "grid":
            return AdjacencyBuilder._build_grid(num_nodes)
        elif topology == "ring":
            return AdjacencyBuilder._build_ring(num_nodes)
        elif topology == "star":
            return AdjacencyBuilder._build_star(num_nodes)
        elif topology == "fully_connected":
            return AdjacencyBuilder._build_fully_connected(num_nodes)
        elif topology == "random":
            return AdjacencyBuilder._build_random(num_nodes)
        else:
            raise ValueError(f"Unknown topology: {topology}")
            
    @staticmethod
    def _build_grid(n: int) -> np.ndarray:
        adj = np.zeros((n, n), dtype=np.float32)
        side = int(np.sqrt(n))
        if side * side != n:
            raise ValueError("Grid topology requires a perfect square number of nodes (e.g. 4, 9, 16)")
            
        for i in range(n):
            row, col = i // side, i % side
            if row > 0: adj[i, i - side] = 1.0 # Up
            if row < side - 1: adj[i, i + side] = 1.0 # Down
            if col > 0: adj[i, i - 1] = 1.0 # Left
            if col < side - 1: adj[i, i + 1] = 1.0 # Right
        return adj
        
    @staticmethod
    def _build_ring(n: int) -> np.ndarray:
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            adj[i, (i - 1) % n] = 1.0
            adj[i, (i + 1) % n] = 1.0
        return adj
        
    @staticmethod
    def _build_star(n: int) -> np.ndarray:
        adj = np.zeros((n, n), dtype=np.float32)
        # Node 0 is the center
        for i in range(1, n):
            adj[0, i] = 1.0
            adj[i, 0] = 1.0
        return adj
        
    @staticmethod
    def _build_fully_connected(n: int) -> np.ndarray:
        adj = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(adj, 0.0)
        return adj
        
    @staticmethod
    def _build_random(n: int) -> np.ndarray:
        adj = np.random.randint(0, 2, size=(n, n)).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        # Make symmetric
        adj = np.maximum(adj, adj.T)
        return adj
