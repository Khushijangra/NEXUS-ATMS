from typing import List
import numpy as np

class RewardSharing:
    """
    Handles the distribution of local vs global reward to prevent independent PPO behavior 
    and encourage cooperative MAPPO routing/spillback prevention.
    """
    def __init__(self, local_weight: float = 0.7, global_weight: float = 0.3):
        self.local_weight = local_weight
        self.global_weight = global_weight
        
    def compute_mixed_rewards(self, local_rewards: List[float], global_metrics: dict) -> List[float]:
        """
        Mixes individual intersection performance with network-wide health.
        global_metrics expects: avg_queue, avg_delay, avg_carbon, emergencies, congestion
        """
        # A simple robust global penalty. Congestion and emergencies are weighted heavily.
        global_penalty = (
            global_metrics["avg_queue"] * 1.0 +
            global_metrics["avg_delay"] * 1.0 +
            global_metrics["avg_carbon"] * 1.0 +
            global_metrics["emergencies"] * 10.0 +
            global_metrics["congestion"] * 2.0
        )
        
        global_reward = -float(global_penalty)
        
        mixed = []
        for lr in local_rewards:
            val = (self.local_weight * lr) + (self.global_weight * global_reward)
            mixed.append(float(val))
            
        return mixed
