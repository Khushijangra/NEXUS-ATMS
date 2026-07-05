import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import time
import numpy as np

def run_mappo_verification():
    print("========================================")
    print("      SPGRL MAPPO SYSTEM VERIFICATION     ")
    print("========================================")
    
    start_time = time.time()
    
    # 1. Imports and Config
    try:
        from v2.rl.mappo.multi_intersection_env import MultiIntersectionEnv
        from v2.rl.mappo.mappo_agent import MAPPOAgent
        from v2.rl.mappo.rollout_buffer import MAPPORolloutBuffer
        from v2.rl.mappo.gae import compute_mappo_gae
        from v2.rl.mappo.learner import MAPPOLearner
        from v2.rl.mappo.communication import MAPPOCommunication
        
        config = {
            'num_agents': 4,
            'topology': 'grid',
            'state_dim': 168,
            'action_dim': 4,
            'critic_type': 'gcn',
            'local_reward_weight': 0.7,
            'global_reward_weight': 0.3,
            'batch_size': 32,
            'epochs': 2
        }
        print("Imports & Configuration      PASS")
    except Exception as e:
        print(f"Imports & Configuration      FAIL ({e})")
        return
        
    # 2. Topology and Adjacency
    try:
        env = MultiIntersectionEnv(config)
        adj = env.get_adjacency()
        assert adj.shape == (4, 4)
        print("Topology & Adjacency         PASS")
    except Exception as e:
        print(f"Topology & Adjacency         FAIL ({e})")
        return
        
    # 3. Environment & Reward Sharing
    try:
        obs_list = env.reset()
        assert len(obs_list) == 4
        assert obs_list[0].shape == (168,)
        
        # Test step
        actions = [np.zeros(4) for _ in range(4)]
        obs_list, rewards, dones, infos = env.step(actions)
        assert len(rewards) == 4
        print("Agents & Reward Sharing      PASS")
    except Exception as e:
        print(f"Agents & Reward Sharing      FAIL ({e})")
        return
        
    # 4. CTDE Networks (Shared Actor & Graph Critic)
    try:
        agent = MAPPOAgent(config)
        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).unsqueeze(0)
        adj_tensor = torch.tensor(adj, dtype=torch.float32).unsqueeze(0)
        global_metrics = torch.tensor(env.get_global_metrics(), dtype=torch.float32).unsqueeze(0)
        
        action, log_prob = agent.act(obs_tensor)
        assert action.shape == (1, 4, 4)
        
        value = agent.evaluate_value(obs_tensor, adj_tensor, global_metrics)
        assert value.shape == (1, 1)
        print("Shared Actor & Graph Critic  PASS")
    except Exception as e:
        print(f"Shared Actor & Graph Critic  FAIL ({e})")
        return
        
    # 5. Communication
    try:
        comm = MAPPOCommunication()
        msgs = comm(obs_tensor, adj_tensor)
        assert msgs.shape == (1, 4, 16)
        print("Communication Layer          PASS")
    except Exception as e:
        print(f"Communication Layer          FAIL ({e})")
        return
        
    # 6. Rollout Buffer
    try:
        buffer = MAPPORolloutBuffer(num_agents=4, buffer_size=10)
        buffer.add(
            obs_tensor.squeeze(0), 
            action.squeeze(0), 
            log_prob.squeeze(0), 
            torch.zeros((4, 1)), 
            torch.zeros((4, 1)), 
            value.squeeze(0), 
            global_metrics.squeeze(0), 
            adj_tensor.squeeze(0),
            msgs.squeeze(0),
            torch.zeros((4, 1))
        )
        assert buffer.step == 1
        print("Rollout Buffer               PASS")
    except Exception as e:
        print(f"Rollout Buffer               FAIL ({e})")
        return
        
    # 7. GAE & Joint Loss
    try:
        adv, ret = compute_mappo_gae(
            buffer.rewards[:1],
            buffer.values[:1],
            buffer.dones[:1],
            value.squeeze(0)
        )
        assert adv.shape == (1, 4, 1)
        
        learner = MAPPOLearner(agent, config)
        actor_loss, val_loss = learner.update(buffer, adv, ret)
        assert isinstance(actor_loss, float)
        print("GAE & Joint PPO Loss         PASS")
    except Exception as e:
        print(f"GAE & Joint PPO Loss         FAIL ({e})")
        return
        
    total_time = time.time() - start_time
    print("========================================")
    print("FINAL STATUS: ALL MAPPO MODULES PASS")
    print(f"Total Latency: {total_time:.2f}s")
    print("========================================")

if __name__ == "__main__":
    run_mappo_verification()
