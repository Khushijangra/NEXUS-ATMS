import numpy as np
from v2.graph.graph_builder import GraphBuilder
def generate_graphs():
    b = GraphBuilder()
    for size in [(1,1), (2,2), (4,4), (8,8)]:
        adj = b.build_grid_graph(*size)
        np.save(f"v2/graph/adjacency_{size[0]*size[1]}.npy", adj)
