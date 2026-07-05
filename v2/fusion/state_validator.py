import numpy as np
import pandas as pd
import json
import time

class UnifiedStateProfiler:
    def profile_tensor(self, name, tensor):
        return {
            "name": name,
            "dimension": tensor.shape,
            "dtype": str(tensor.dtype),
            "memory_bytes": tensor.nbytes,
            "nan_count": int(np.isnan(tensor).sum()),
            "inf_count": int(np.isinf(tensor).sum()),
            "variance": float(np.var(tensor)),
            "mean": float(np.mean(tensor)),
            "std": float(np.std(tensor)),
            "min": float(np.min(tensor)),
            "max": float(np.max(tensor)),
            "sparsity": float(np.sum(tensor == 0) / tensor.size) if tensor.size > 0 else 0
        }

    def generate_profiles(self):
        # Mock Zt components
        Gt = np.random.randn(10, 10)
        As = np.random.randn(10)
        Ab = np.random.randn(10)
        Ft = np.random.randn(10)
        Cf = np.random.randn(1)
        Ct = np.random.randn(1)
        Et = np.random.randn(1)
        
        Zt_components = {"Gt": Gt, "As": As, "Ab": Ab, "Ft": Ft, "Cf": Cf, "Ct": Ct, "Et": Et}
        
        start = time.perf_counter()
        Zt = np.concatenate([c.flatten() for c in Zt_components.values()])
        latency = (time.perf_counter() - start) * 1000 # ms
        
        profiles = [self.profile_tensor(k, v) for k, v in Zt_components.items()]
        profiles.append(self.profile_tensor("Zt_total", Zt))
        
        with open("v2/fusion/Zt_profile.json", "w") as f:
            json.dump(profiles, f, indent=4)
            
        df_mem = pd.DataFrame([{"component": p["name"], "memory_bytes": p["memory_bytes"]} for p in profiles])
        df_mem.to_csv("v2/fusion/Zt_memory.csv", index=False)
        
        df_lat = pd.DataFrame([{"operation": "Zt_concatenation", "latency_ms": latency}])
        df_lat.to_csv("v2/fusion/Zt_latency.csv", index=False)

if __name__ == "__main__":
    profiler = UnifiedStateProfiler()
    profiler.generate_profiles()
