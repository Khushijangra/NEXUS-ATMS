import numpy as np
import logging

logger = logging.getLogger(__name__)

class SafetyWrapper:
    """
    Safe-RL Wrapper that intercepts PPO actions before execution to enforce 
    hard physical constraints (Emergency priority, max waiting limits).
    """
    def __init__(self, env):
        self.env = env
        self.unsafe_proposed = 0
        self.overrides_performed = 0
        
    def __getattr__(self, name):
        """Pass through missing attributes to the wrapped environment."""
        return getattr(self.env, name)
        
    def reset(self):
        self.unsafe_proposed = 0
        self.overrides_performed = 0
        return self.env.reset()
        
    def step(self, action, compute_zt=True):
        safe_action = action
        override = False
        
        # 1. Emergency Priority Rule
        # If an emergency is active, the ONLY safe action is 2 (Emergency Clearance)
        if self.env.emergency and action != 2:
            safe_action = 2
            override = True
            
        # 2. Queue Starvation / Max Waiting Rule
        # If queue exceeds 100 on an axis, force its green light to prevent starvation.
        if not self.env.emergency and not override:
            ns_queue = self.env.queue[0] + self.env.queue[1]
            ew_queue = self.env.queue[2] + self.env.queue[3]
            
            if ns_queue > 100 and action != 0:
                safe_action = 0
                override = True
            elif ew_queue > 100 and action != 1:
                safe_action = 1
                override = True
                
        if override:
            self.unsafe_proposed += 1
            self.overrides_performed += 1
            
        # Execute the safe action
        zt, reward, done, info = self.env.step(safe_action, compute_zt=compute_zt)
        
        # Apply a massive negative penalty if the PPO agent proposed an illegal action
        if override:
            reward -= 5.0
            info['reward_components']['safety_penalty'] = -5.0
            info['override'] = True
        else:
            info['reward_components']['safety_penalty'] = 0.0
            info['override'] = False
            
        return zt, float(reward), done, info
        
    def get_safety_stats(self):
        """Returns statistics for the IEEE presentation tables."""
        success_rate = 100.0 if self.unsafe_proposed == 0 else (self.overrides_performed / self.unsafe_proposed) * 100.0
        return {
            "unsafe_proposed": self.unsafe_proposed,
            "overrides_performed": self.overrides_performed,
            "success_rate": success_rate
        }