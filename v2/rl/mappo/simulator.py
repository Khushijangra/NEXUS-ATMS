import numpy as np

class TrafficNetworkSimulator:
    """
    Simulates the physical dynamics of a multi-intersection traffic network,
    including spillback, vehicle movement between nodes, and emergency routing.
    """
    def __init__(self, num_intersections: int, adjacency_matrix: np.ndarray):
        self.num_nodes = num_intersections
        self.adjacency = adjacency_matrix
        
        # Core network state
        self.queues = np.zeros(self.num_nodes, dtype=np.float32)
        self.delays = np.zeros(self.num_nodes, dtype=np.float32)
        self.carbons = np.zeros(self.num_nodes, dtype=np.float32)
        self.emergencies = np.zeros(self.num_nodes, dtype=np.int32)
        
        # Link capacities (max vehicles a link can hold before spillback)
        self.link_capacities = np.ones((self.num_nodes, self.num_nodes)) * 50.0
        self.link_vehicles = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
    def reset(self):
        self.queues.fill(0)
        self.delays.fill(0)
        self.carbons.fill(0)
        self.emergencies.fill(0)
        self.link_vehicles.fill(0)
        return self._get_network_state()
        
    def step(self, local_actions: np.ndarray, base_arrivals: np.ndarray):
        """
        Step the network forward.
        local_actions: [N] array of phase durations or green/red choices.
        base_arrivals: [N] exogenous vehicle arrivals per intersection.
        """
        # 1. Process exogenous arrivals
        self.queues += base_arrivals
        
        # 2. Process intersection throughput (based on local actions)
        # Simplified: Green phase reduces queue by 'throughput_rate'
        throughput = np.where(local_actions > 0, 10.0, 2.0)
        cleared_vehicles = np.minimum(self.queues, throughput)
        self.queues -= cleared_vehicles
        
        # 3. Network Movement & Spillback
        # Vehicles cleared from Node i distribute to neighbors j
        for i in range(self.num_nodes):
            neighbors = np.where(self.adjacency[i] > 0)[0]
            if len(neighbors) > 0 and cleared_vehicles[i] > 0:
                # Distribute evenly to neighbors
                flow_per_neighbor = cleared_vehicles[i] / len(neighbors)
                for j in neighbors:
                    # Check spillback
                    available_space = self.link_capacities[i, j] - self.link_vehicles[i, j]
                    actual_flow = min(flow_per_neighbor, available_space)
                    
                    self.link_vehicles[i, j] += actual_flow
                    # If spillback occurs, penalty to Node i's queue
                    spillback = flow_per_neighbor - actual_flow
                    self.queues[i] += spillback
                    
        # 4. Vehicles arriving from links to next intersection's queue
        arrival_rate = 0.5 # 50% of vehicles on link arrive at next node per step
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.link_vehicles[i, j] > 0:
                    arriving = self.link_vehicles[i, j] * arrival_rate
                    self.link_vehicles[i, j] -= arriving
                    self.queues[j] += arriving
                    
        # 5. Update Delays, Carbon, Emergency
        self.delays += self.queues * 0.1
        self.carbons = self.queues * 0.05
        # Random emergency spawn (1% chance per node)
        self.emergencies = np.random.binomial(1, 0.01, self.num_nodes)
        
        return self._get_network_state()
        
    def _get_network_state(self):
        return {
            "queues": self.queues.copy(),
            "delays": self.delays.copy(),
            "carbons": self.carbons.copy(),
            "emergencies": self.emergencies.copy()
        }
