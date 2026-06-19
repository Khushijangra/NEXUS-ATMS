import time
import subprocess
import os
import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# Make sure we can import everything
sys.path.insert(0, os.path.abspath("."))

from ai.envs.env_anomaly import AnomalySumoEnvironment
from ai.rl.d3qn_multimodal import D3QNAgent

def run_smoke_test():
    server_process = None
    try:
        # 1. Start inference server in dummy mode
        logging.info("Starting inference_server.py...")
        server_cmd = [sys.executable, "argus_stream_extracted/argus stream A/scripts/inference_server.py", "--dummy"]
        server_process = subprocess.Popen(server_cmd)
        time.sleep(2) # Wait for server to bind

        # 2. Instantiate env
        logging.info("Instantiating AnomalySumoEnvironment...")
        env = AnomalySumoEnvironment(
            net_file="networks/single_intersection.net.xml",
            route_file="networks/single_intersection.rou.xml",
            use_gui=False,
            max_steps=200,
            incident_prob=0.05
        )
        
        # Check observation space
        assert env.observation_space.shape[0] == env.state_dim, "Observation space shape mismatch"

        # 3. Instantiate Agent
        logging.info("Instantiating D3QNAgent...")
        config = {
            "agent": {
                "device": "cpu",
                "d3qn": {
                    "batch_size": 32,
                    "train_freq": 1,
                    "buffer_size": 1000
                }
            },
            "training": {"seed": 42, "deterministic": True}
        }
        
        agent = D3QNAgent(env=env, config=config, log_dir="logs_smoke", model_dir="models_smoke")
        
        # Verify agent state dim matches env
        assert agent.state_dim == env.state_dim, "Agent state_dim mismatch"
        
        logging.info(f"Verified State Dim: {agent.state_dim} (N+2)")

        # 4. Run loop
        obs, info = env.reset()
        assert len(obs) == agent.state_dim, "Reset obs shape mismatch"
        
        logging.info("Starting simulation loop...")
        for i in range(150):
            action = agent.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Verify dimensions
            assert len(next_obs) == agent.state_dim, f"Step obs shape mismatch at step {i}"
            
            # Replay buffer check
            agent.replay_buffer.push(
                agent._flatten_obs(obs),
                action,
                reward,
                agent._flatten_obs(next_obs),
                float(terminated or truncated)
            )
            
            obs = next_obs
            
            if step_info.get("incident_type") != "none":
                logging.info(f"Step {i}: Incident injected - {step_info['incident_type']} | Anomaly Score: {step_info['anomaly_score']:.3f} | Flag: {step_info['anomaly_flag']}")

            if terminated or truncated:
                obs, info = env.reset()
                
        # Buffer check
        assert len(agent.replay_buffer) == 150, "Replay buffer did not collect transitions correctly"
        
        # Check optimization
        stats = agent._optimize_step()
        assert stats is not None, "Optimization step failed"
        logging.info("Optimization step succeeded. Smoke test PASSED.")

    except Exception as e:
        logging.error(f"Smoke test FAILED: {e}")
        raise
    finally:
        if server_process:
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    run_smoke_test()
