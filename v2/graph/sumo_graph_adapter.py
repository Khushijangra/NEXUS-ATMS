import numpy as np

class SumoGraphAdapter:
    def __init__(self, traci_connection):
        self.traci = traci_connection
        self.junctions = []
        self.lanes = []
        
    def extract_Xt(self):
        # Extract queue, wait, occupancy, speed, throughput
        # This is a structural skeleton compliant with V1 freeze
        
        num_nodes = len(self.junctions) if self.junctions else 16
        Xt = np.zeros((num_nodes, 8)) 
        
        # Xt[:, 0] = queue
        # Xt[:, 1] = wait
        # Xt[:, 2] = occupancy
        # Xt[:, 3] = speed
        # Xt[:, 4] = throughput
        # Xt[:, 5] = neighbor_queues
        # Xt[:, 6] = neighbor_waits
        # Xt[:, 7] = neighbor_throughput
        
        return Xt
