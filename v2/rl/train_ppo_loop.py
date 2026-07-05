import sys
from pathlib import Path
import logging
import numpy as np
import torch
import time

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.rl.spgrl_environment import SPGRLEnv
from v2.rl.ppo_agent import PPOAgent, RolloutBuffer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import csv

def train(manager, max_episodes=500, resume=False):
    logger = manager.logger
    logger.info("Initializing SPGRL Environment and PPO Agent...")
    env = SPGRLEnv()
    agent = PPOAgent(state_dim=168, action_dim=4)
    buffer = RolloutBuffer()
    
    update_timestep = 200 # Update every full episode
    
    start_episode = 1
    time_step = 0
    running_reward = 0
    best_avg_reward = -float('inf')
    best_queue = float('inf')
    
    ckpt_dir = manager.current_exp_dir / "checkpoints"
    latest_actor = ckpt_dir / "latest_actor.pth"
    latest_critic = ckpt_dir / "latest_critic.pth"
    
    if resume and latest_actor.exists() and latest_critic.exists():
        logger.info(f"Resuming from {latest_actor}")
        agent.actor.load_state_dict(torch.load(latest_actor, weights_only=True))
        agent.critic.load_state_dict(torch.load(latest_critic, weights_only=True))
        
        # Load RNG state if exists
        rng_path = ckpt_dir / "rng_state.pt"
        if rng_path.exists():
            rng_data = torch.load(rng_path, weights_only=False)
            torch.set_rng_state(rng_data['torch'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_data['cuda'])
            np.random.set_state(rng_data['numpy'])
            import random
            random.setstate(rng_data['python'])
            start_episode = rng_data['episode'] + 1
            running_reward = rng_data['running_reward']
            best_avg_reward = rng_data['best_avg_reward']
            logger.info(f"Resumed at episode {start_episode}")
            
    logger.info("Starting End-to-End PPO Training Loop")
    logger.info("=" * 50)
    
    for i_episode in range(start_episode, max_episodes + 1):
        state = env.reset()
        ep_reward = 0
        done = False
        
        last_loss = 0.0
        last_entropy = 0.0
        
        safety_overrides = 0
        start_time = time.time()
        
        while not done:
            action = agent.act(state, buffer)
            
            next_state, reward, done, info = env.step(action)
            
            # Count safety overrides if any logic provides it in info
            # For now mock it as 0 since it's not strictly returned by step yet
            safety_overrides += info.get('safety_overrides', 0)
            
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)
            
            state = next_state
            ep_reward += reward
            time_step += 1
            
            if time_step % update_timestep == 0:
                last_loss, last_entropy = agent.update(buffer)
                
        end_time = time.time()
        inf_time = end_time - start_time
        
        running_reward += ep_reward
        avg_reward = running_reward / i_episode
        
        queue_val = float(np.sum(env.queue))
        delay_val = float(np.sum(env.wait))
        carbon_val = float(env.carbon)
        
        # CSV Logging via manager
        manager.log_metrics_row(
            episode=i_episode, 
            reward=ep_reward, 
            queue=queue_val, 
            delay=delay_val, 
            carbon=carbon_val, 
            entropy=last_entropy, 
            policy_loss=last_loss, 
            value_loss=0.0, # Placeholder if separate value loss is not tracked
            time=inf_time
        )
            
        logger.info(f"Episode {i_episode:03d} | Reward: {ep_reward:7.1f} | Queue: {queue_val:6.1f} | Loss: {last_loss:7.4f} | Time: {inf_time:5.1f}s")
        
        # Save checkpoints
        torch.save(agent.actor.state_dict(), latest_actor)
        torch.save(agent.critic.state_dict(), latest_critic)
        
        # Save RNG states
        import random
        rng_data = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': np.random.get_state(),
            'python': random.getstate(),
            'episode': i_episode,
            'running_reward': running_reward,
            'best_avg_reward': best_avg_reward
        }
        torch.save(rng_data, ckpt_dir / "rng_state.pt")
        
        if avg_reward > best_avg_reward and i_episode > 5:
            best_avg_reward = avg_reward
            torch.save(agent.actor.state_dict(), ckpt_dir / "best_reward_actor.pth")
            
        if queue_val < best_queue and i_episode > 5:
            best_queue = queue_val
            torch.save(agent.actor.state_dict(), ckpt_dir / "best_queue_actor.pth")
            
        if delay_val < getattr(manager, 'best_delay', float('inf')) and i_episode > 5:
            manager.best_delay = delay_val
            torch.save(agent.actor.state_dict(), ckpt_dir / "best_delay_actor.pth")
            
        if carbon_val < getattr(manager, 'best_carbon', float('inf')) and i_episode > 5:
            manager.best_carbon = carbon_val
            torch.save(agent.actor.state_dict(), ckpt_dir / "best_carbon_actor.pth")
            
    # Final summary
    summary = {
        "Total_Episodes": max_episodes,
        "Best_Avg_Reward": float(best_avg_reward),
        "Best_Queue": float(best_queue),
        "Best_Delay": float(getattr(manager, 'best_delay', 0.0)),
        "Best_Carbon": float(getattr(manager, 'best_carbon', 0.0))
    }
    manager.write_summary(summary)
    logger.info("Training complete. Data provenance and metrics logged successfully.")

