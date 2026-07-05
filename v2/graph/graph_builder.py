import numpy as np
import networkx as nx
import json

class GridGraphGenerator:
    def _build_grid(self, rows, cols):
        G = nx.grid_2d_graph(rows, cols)
        # Convert nodes to integers
        G = nx.convert_node_labels_to_integers(G)
        return G
        
    def build_1x1(self): return self._build_grid(1, 1)
    def build_2x2(self): return self._build_grid(2, 2)
    def build_4x4(self): return self._build_grid(4, 4)
    def build_8x8(self): return self._build_grid(8, 8)
    
    def compute_metrics(self, G):
        A = nx.adjacency_matrix(G).todense()
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        
        cent = nx.degree_centrality(G)
        betw = nx.betweenness_centrality(G)
        clust = nx.clustering(G)
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        
        # Convert path lengths to matrix S
        n = len(G.nodes)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = path_lengths.get(i, {}).get(j, 0)
                
        return A, D, L, cent, betw, clust, S

def generate_graphs():
    gen = GridGraphGenerator()
    G = gen.build_4x4()
    A, D, L, cent, betw, clust, S = gen.compute_metrics(G)
    
    np.save("v2/graph/graph_adjacency.npy", np.array(A))
    np.save("v2/graph/graph_laplacian.npy", np.array(L))
    
    with open("v2/graph/graph_statistics.json", "w") as f:
        json.dump({"nodes": len(G.nodes), "edges": len(G.edges)}, f)
        
    import pandas as pd
    df = pd.DataFrame({"node": list(cent.keys()), "degree_centrality": list(cent.values()), 
                       "betweenness": list(betw.values()), "clustering": list(clust.values())})
    df.to_csv("v2/graph/graph_centrality.csv", index=False)

if __name__ == "__main__":
    generate_graphs()
