import sys
from pathlib import Path
import logging
import numpy as np
import torch
import collections

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.rl.spgrl_environment import SPGRLEnv
from v2.rl.ppo_agent import PPOAgent
from v2.rl.ppo_agent import RolloutBuffer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_10_episodes():
    env = SPGRLEnv()
    agent = PPOAgent(state_dim=168, action_dim=4)
    buffer = RolloutBuffer()
    
    update_timestep = 200
    time_step = 0
    
    logger.info("Starting 10-Episode MDP Verification Loop")
    logger.info("=" * 50)
    
    for i_episode in range(1, 11):
        state = env.reset()
        done = False
        
        ep_reward = 0
        action_counts = collections.Counter()
        
        # For logging Et transitions
        prev_et = state[-1] # Et is the last element
        
        while not done:
            action = agent.act(state, buffer)
            action_counts[action] += 1
            
            next_state, reward, done, info = env.step(action)
            
            curr_et = next_state[-1]
            
            # Log Et transitions
            if prev_et != curr_et:
                logger.info(f"[Ep {i_episode:02d} Step {env.current_step:03d}] Et Transition: {prev_et:.1f} -> {curr_et:.1f} | Action: {action}")
                
            # Log queue evolution on specific steps
            if env.current_step in [10, 50, 100, 150, 200]:
                q_total = np.sum(env.queue)
                comp = info.get('reward_components', {})
                logger.info(f"  [Step {env.current_step:03d}] Queue: {q_total:6.1f} | Et: {curr_et:.1f} | Action: {action} | Raw Reward: {comp.get('raw_reward', 0.0):.2f}")
                
            prev_et = curr_et
            buffer.rewards.append(reward)
            buffer.is_terminals.append(done)
            
            state = next_state
            ep_reward += reward
            time_step += 1
            
            if time_step % update_timestep == 0:
                agent.update(buffer)
                
        logger.info(f"Episode {i_episode:02d} Summary:")
        logger.info(f"  Action Distribution: {dict(action_counts)}")
        logger.info(f"  Final Reward: {ep_reward:.2f}\n")
        
if __name__ == '__main__':
    verify_10_episodes()
