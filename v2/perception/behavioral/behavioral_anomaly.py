import numpy as np
import json
import pandas as pd

class BehavioralAnomalyEngine:
    def __init__(self, mu_v=15.0, sigma_v=5.0, mu_a=0.0, sigma_a=2.0, epsilon=1e-8):
        self.mu_v = mu_v
        self.sigma_v = sigma_v
        self.mu_a = mu_a
        self.sigma_a = sigma_a
        self.epsilon = epsilon
        
        self.weights = {
            "speed": 0.30,
            "acceleration": 0.25,
            "jerk": 0.20,
            "entropy": 0.15,
            "wrong_way": 0.10
        }
        
    def speed_anomaly(self, v):
        return (v - self.mu_v) / (self.sigma_v + self.epsilon)
        
    def acceleration_anomaly(self, a):
        return (a - self.mu_a) / (self.sigma_a + self.epsilon)
        
    def jerk(self, a_t, a_t_minus_1):
        return a_t - a_t_minus_1
        
    def entropy(self, probabilities):
        p = np.array(probabilities)
        return -np.sum(p * np.log(p + self.epsilon))
        
    def wrong_way(self, theta, theta_thr=150):
        return 1 if theta > theta_thr else 0
        
    def compute_Ab(self, v, a_t, a_t_minus_1, probabilities, theta):
        zv = self.speed_anomaly(v)
        za = self.acceleration_anomaly(a_t)
        jt = self.jerk(a_t, a_t_minus_1)
        H = self.entropy(probabilities)
        W = self.wrong_way(theta)
        
        Ab = (self.weights["speed"] * zv + 
              self.weights["acceleration"] * za + 
              self.weights["jerk"] * jt + 
              self.weights["entropy"] * H + 
              self.weights["wrong_way"] * W)
        return Ab

def generate_mock_outputs():
    # Generate mock Ab matrix for scaffolding validation
    Ab = np.random.randn(10, 10)
    np.save("v2/perception/behavioral/Ab.npy", Ab)
    
    with open("v2/perception/behavioral/behavioral_statistics.json", "w") as f:
        json.dump({"mean_Ab": float(np.mean(Ab)), "std_Ab": float(np.std(Ab))}, f)
        
    df = pd.DataFrame({"timestamp": range(100), "veh_id": np.random.randint(0, 50, 100), "Ab": np.random.randn(100)})
    df.to_csv("v2/perception/behavioral/behavioral_traceability.csv", index=False)

if __name__ == "__main__":
    generate_mock_outputs()
