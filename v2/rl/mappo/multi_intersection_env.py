import numpy as np
from typing import Dict, Tuple, List
from v2.rl.spgrl_environment import SPGRLEnv
from v2.rl.mappo.simulator import TrafficNetworkSimulator
from v2.rl.mappo.adjacency import AdjacencyBuilder
from v2.rl.mappo.reward_sharing import RewardSharing
import copy

class MultiIntersectionEnv:
    """
    MAPPO wrapper that manages N independent SPGRLEnv instances 
    coupled physically by the TrafficNetworkSimulator.
    """
    def __init__(self, config: dict):
        self.config = config
        self.num_agents = config.get('num_agents', 4)
        self.topology = config.get('topology', 'grid')
        
        # Build Adjacency Matrix
        self.adjacency = AdjacencyBuilder.build(self.topology, self.num_agents)
        
        # Initialize N SPGRL single-agent environments (they handle the 168D building)
        # We override their internal queues based on the Simulator
        self.agents = [SPGRLEnv() for _ in range(self.num_agents)]
        
        # Initialize Network Simulator
        self.simulator = TrafficNetworkSimulator(self.num_agents, self.adjacency)
        
        self.local_reward_weight = config.get('local_reward_weight', 0.7)
        self.global_reward_weight = config.get('global_reward_weight', 0.3)
        self.reward_sharer = RewardSharing(self.local_reward_weight, self.global_reward_weight)
        
        # Track global metrics
        self.global_metrics = {
            "avg_queue": 0.0,
            "avg_delay": 0.0,
            "avg_carbon": 0.0,
            "emergencies": 0,
            "congestion": 0.0
        }
        
    def reset(self) -> List[np.ndarray]:
        sim_state = self.simulator.reset()
        observations = []
        
        for i, agent in enumerate(self.agents):
            obs = agent.reset()
            # Override SPGRL's queue/wait with simulator's coupled reality
            # SPGRLEnv expects shape (4,) for queue and wait, but simulator provides scalar per intersection.
            # We will map the scalar uniformly across the 4 directions for SPGRLEnv's internal logic.
            agent.queue = np.full(4, sim_state['queues'][i] / 4.0, dtype=np.float32)
            agent.wait = np.full(4, sim_state['delays'][i] / 4.0, dtype=np.float32)
            agent.carbon = sim_state['carbons'][i]
            # Re-build the 168D state so it reflects the new physical reality
            obs = agent._build_zt()
            observations.append(obs)
            
        return observations
        
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], List[dict]]:
        """
        Executes a joint step across the multi-agent environment.
        actions: list of [4] discrete/continuous actions per agent
        """
        # 1. Gather baseline arrivals from each independent SPGRLEnv
        # (This pretends they are independent just to get the base flow)
        base_arrivals = np.zeros(self.num_agents, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            # We don't call agent.step() directly yet because that updates its isolated state
            # We just get the exogenous demand from its internal step logic generator
            # For this simplified proxy, we'll just inject random arrivals
            base_arrivals[i] = np.random.poisson(2.0)
            
        # 2. Step the physical simulator
        # We need a scalar action for the simulator throughput (e.g. green time)
        scalar_actions = np.array([np.argmax(a) if len(a) > 1 else a[0] for a in actions])
        sim_state = self.simulator.step(scalar_actions, base_arrivals)
        
        observations = []
        local_rewards = []
        dones = []
        infos = []
        
        # 3. Synchronize individual environments & calculate local rewards
        for i, agent in enumerate(self.agents):
            # Sync physics
            agent.queue = np.full(4, sim_state['queues'][i] / 4.0, dtype=np.float32)
            agent.wait = np.full(4, sim_state['delays'][i] / 4.0, dtype=np.float32)
            agent.carbon = sim_state['carbons'][i]
            
            # Action semantics from SPGRLEnv
            act = scalar_actions[i]
            
            # Replicate local reward logic from SPGRLEnv
            Q_max = 200.0
            D_max = 1000.0
            C_max = 1000.0
            congestion_penalty = np.sum(agent.queue) / Q_max
            delay_penalty = np.sum(agent.wait) / D_max
            carbon_penalty = agent.carbon / C_max
            
            emergency_reward = 2.0 if (act == 2 and agent.emergency) else 0.0
            if agent.emergency and act != 2:
                emergency_reward = -2.0 # Missing emergency penalty
                
            anomaly_norm = np.tanh(abs(agent.anomaly_score) / 100.0)
            raw_reward = emergency_reward - congestion_penalty - delay_penalty - carbon_penalty - anomaly_norm
            local_reward = 0.9 * raw_reward + 0.1 * agent.prev_reward
            agent.prev_reward = local_reward
            
            # Step internal counters
            agent.current_step += 1
            done = agent.current_step >= agent.max_steps
            
            # Build new observation
            obs = agent._build_zt()
            
            observations.append(obs)
            local_rewards.append(local_reward)
            dones.append(done)
            infos.append({
                'safety_overrides': 0, # Integrate joint safety later
                'queue': sim_state['queues'][i],
                'delay': sim_state['delays'][i],
                'carbon': sim_state['carbons'][i]
            })
            
        # 4. Calculate Global Metrics & Reward Sharing
        self.global_metrics["avg_queue"] = np.mean(sim_state['queues'])
        self.global_metrics["avg_delay"] = np.mean(sim_state['delays'])
        self.global_metrics["avg_carbon"] = np.mean(sim_state['carbons'])
        self.global_metrics["emergencies"] = np.sum(sim_state['emergencies'])
        self.global_metrics["congestion"] = np.sum(self.simulator.link_vehicles)
        
        # Normalize global reward (simple proxy: negative of averages)
        # 5. Distribute Mixed Reward
        mixed_rewards = self.reward_sharer.compute_mixed_rewards(local_rewards, self.global_metrics)
            
        return observations, mixed_rewards, dones, infos
        
    def get_global_metrics(self) -> np.ndarray:
        """Returns [5] global metrics for the Centralized Critic."""
        return np.array([
            self.global_metrics["avg_queue"],
            self.global_metrics["avg_delay"],
            self.global_metrics["avg_carbon"],
            float(self.global_metrics["emergencies"]),
            self.global_metrics["congestion"]
        ], dtype=np.float32)
        
    def get_adjacency(self) -> np.ndarray:
        return self.adjacency
