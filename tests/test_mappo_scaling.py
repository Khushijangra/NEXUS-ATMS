import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from v2.rl.mappo.multi_intersection_env import MultiIntersectionEnv
from v2.rl.mappo.critic import CentralizedCritic
from v2.rl.mappo.actor import SharedActor
from v2.rl.mappo.communication import MAPPOCommunication
import torch

def test_mappo_scaling(n_agents: int, topology: str):
    print(f"\n--- Testing N={n_agents} with {topology} topology ---")
    config = {
        'num_agents': n_agents,
        'topology': topology,
        'local_reward_weight': 0.7,
        'global_reward_weight': 0.3
    }
    
    # 1. Env
    env = MultiIntersectionEnv(config)
    obs = env.reset()
    assert len(obs) == n_agents
    assert obs[0].shape == (168,)
    print(f"[OK] Env Reset: Got {len(obs)} observations.")
    
    adj = env.get_adjacency()
    assert adj.shape == (n_agents, n_agents)
    print(f"[OK] Adjacency Shape: {adj.shape}")
    
    # 2. Shared Actor
    actor = SharedActor()
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    action, logprob = actor.get_action(obs_tensor)
    assert action.shape == (n_agents, 4)
    print(f"[OK] Shared Actor: Action shape {action.shape}")
    
    # 3. Communication
    comm = MAPPOCommunication(msg_dim=16)
    adj_tensor = torch.tensor(adj, dtype=torch.float32).unsqueeze(0) # [1, N, N]
    obs_batched = obs_tensor.unsqueeze(0) # [1, N, 168]
    msgs = comm(obs_batched, adj_tensor)
    assert msgs.shape == (1, n_agents, 16)
    print(f"[OK] Communication: Message shape {msgs.shape}")
    
    # 4. Centralized Critic
    critic = CentralizedCritic()
    global_metrics = torch.tensor(env.get_global_metrics(), dtype=torch.float32).unsqueeze(0)
    val = critic(obs_batched, adj_tensor, global_metrics)
    assert val.shape == (1, 1)
    print(f"[OK] Centralized Critic: Value shape {val.shape}")

if __name__ == "__main__":
    test_mappo_scaling(n_agents=2, topology="grid")
    test_mappo_scaling(n_agents=4, topology="fully_connected")
    test_mappo_scaling(n_agents=8, topology="star")
    test_mappo_scaling(n_agents=16, topology="ring")
    print("\nAll MAPPO scaling architectures pass verification!")
