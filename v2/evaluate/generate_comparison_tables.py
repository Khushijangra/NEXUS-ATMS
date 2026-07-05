import sys
from pathlib import Path
import numpy as np
import torch
import csv

root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from v2.rl.spgrl_environment import SPGRLEnv
from v2.rl.ppo_agent import PPOAgent

def evaluate_controller(env, agent, controller_type, num_episodes=100):
    total_rewards = []
    final_queues = []
    final_delays = []
    final_carbons = []
    emergency_clearances = 0
    total_emergencies = 0
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        ep_reward = 0
        
        ep_emergency_cleared = False
        ep_had_emergency = env.emergency
        
        while not done:
            was_emergency = env.emergency
            
            if controller_type == "random":
                action = np.random.randint(0, env.action_space_dim)
            elif controller_type == "fixed":
                action = step % env.action_space_dim
            elif controller_type == "actuated":
                q_ns = np.sum(env.queue[[0, 1]])
                q_ew = np.sum(env.queue[[2, 3]])
                if env.emergency:
                    action = 2
                elif q_ns > q_ew:
                    action = 0
                elif q_ew > q_ns:
                    action = 1
                else:
                    action = 3
            elif controller_type == "ppo":
                # Deterministic evaluation (no buffer needed)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_probs = agent.policy_old(state_tensor)
                    action = torch.argmax(action_probs, dim=-1).item()
                    
            compute_zt_flag = True if controller_type == "ppo" else False
            state, reward, done, _ = env.step(action, compute_zt=compute_zt_flag)
            ep_reward += reward
            step += 1
            
            # Check if emergency was cleared this step
            if was_emergency and not env.emergency:
                ep_emergency_cleared = True
                
        total_rewards.append(ep_reward)
        final_queues.append(np.sum(env.queue))
        final_delays.append(np.sum(env.wait))
        final_carbons.append(env.carbon)
        
        if ep_had_emergency:
            total_emergencies += 1
            if ep_emergency_cleared:
                emergency_clearances += 1
                
    clearance_rate = (emergency_clearances / total_emergencies * 100) if total_emergencies > 0 else 100.0
        
    return {
        "reward": np.mean(total_rewards),
        "queue": np.mean(final_queues),
        "delay": np.mean(final_delays),
        "carbon": np.mean(final_carbons),
        "clearance_rate": clearance_rate
    }

def main():
    print("Loading Best PPO Agent...")
    env = SPGRLEnv()
    agent = PPOAgent(state_dim=168, action_dim=4)
    
    actor_path = Path(root) / "models" / "spgrl" / "best_actor.pth"
    if actor_path.exists():
        agent.policy_old.load_state_dict(torch.load(actor_path))
        print("Model loaded successfully.")
    else:
        print("No trained model found! Please ensure training has run.")
        return
        
    results = {}
    controllers = ["random", "fixed", "actuated", "ppo"]
    
    for ctrl in controllers:
        print(f"Evaluating {ctrl.upper()} Controller (100 episodes)...")
        res = evaluate_controller(env, agent, ctrl, num_episodes=100)
        results[ctrl] = res
        
    # Print Markdown Table
    print("\n--- Final 100-Episode Comparison Benchmark ---")
    print(f"| Controller | Avg Reward | Queue | Delay | Carbon | Emergency Clearance |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
    for ctrl in controllers:
        res = results[ctrl]
        name = ctrl.capitalize() if ctrl != 'ppo' else 'SPGRL PPO'
        print(f"| {name} | {res['reward']:.2f} | {res['queue']:.2f} | {res['delay']:.2f} | {res['carbon']:.2f} | {res['clearance_rate']:.1f}% |")
        
    # Save CSV
    csv_file = Path(root) / "v2" / "evaluate" / "comparison_tables.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Controller", "Avg Reward", "Queue", "Delay", "Carbon", "Emergency Clearance Rate"])
        for ctrl in controllers:
            res = results[ctrl]
            name = ctrl.capitalize() if ctrl != 'ppo' else 'SPGRL PPO'
            writer.writerow([name, res['reward'], res['queue'], res['delay'], res['carbon'], res['clearance_rate']])
            
    print(f"\nSaved CSV to {csv_file}")

if __name__ == "__main__":
    main()
