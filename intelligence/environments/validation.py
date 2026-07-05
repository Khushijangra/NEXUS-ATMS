import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class EnvironmentValidator:
    """
    Mandatory pre-flight validation of the RL environment.
    Runs 100 deterministic steps to ensure observation shapes, finite tensors, and no NaNs/Infs.
    """
    
    @staticmethod
    def validate(env, output_dir: Path, num_steps: int = 100) -> bool:
        logger.info(f"Running Environment Pre-Flight Validation ({num_steps} steps)...")
        errors = []
        
        try:
            obs, info = env.reset(seed=42)
            expected_shape = env.observation_space.shape
            
            for step in range(num_steps):
                # 1. Check Shape
                if obs.shape != expected_shape:
                    errors.append(f"Step {step}: Shape mismatch. Expected {expected_shape}, got {obs.shape}")
                    break
                    
                # 2. Check Finite and NaNs
                if not np.isfinite(obs).all():
                    errors.append(f"Step {step}: Observation contains NaN or Inf.")
                    break
                    
                # Take random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Check reward
                if not np.isfinite(reward):
                    errors.append(f"Step {step}: Reward contains NaN or Inf: {reward}")
                    break
                    
                if terminated or truncated:
                    obs, info = env.reset()
                    
        except Exception as e:
            errors.append(f"Exception during environment step: {str(e)}")
            
        is_valid = len(errors) == 0
        report = {
            "status": "PASS" if is_valid else "FAIL",
            "steps_executed": num_steps,
            "expected_shape": list(expected_shape) if 'expected_shape' in locals() else None,
            "errors": errors
        }
        
        report_path = output_dir / "environment_validation.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        if not is_valid:
            logger.error("Environment Validation Failed!")
            for err in errors:
                logger.error(f"  [X] {err}")
        else:
            logger.info("Environment Validation Passed. Ready for training.")
            
        return is_valid
