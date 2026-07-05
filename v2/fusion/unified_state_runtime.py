import numpy as np
import json
import time

def build_unified_state(Gt, As, Ab, Ft, Cf, Ct, Et):
    # Zt = [Gt, As, Ab, Ft, Cf, Ct, Et]
    start = time.perf_counter()
    Zt = np.concatenate([
        Gt.flatten(), 
        As.flatten(), 
        Ab.flatten(), 
        Ft.flatten(), 
        Cf.flatten(), 
        Ct.flatten(), 
        Et.flatten()
    ])
    latency = (time.perf_counter() - start) * 1000 # ms
    
    dim_report = {
        "Gt": Gt.shape, "As": As.shape, "Ab": Ab.shape, "Ft": Ft.shape,
        "Cf": Cf.shape, "Ct": Ct.shape, "Et": Et.shape, "Zt": Zt.shape
    }
    
    with open("v2/fusion/zt_dimension_report.json", "w") as f:
        json.dump(dim_report, f)
        
    with open("v2/fusion/zt_memory_profile.json", "w") as f:
        json.dump({"Zt_memory_bytes": Zt.nbytes}, f)
        
    with open("v2/fusion/zt_latency_profile.json", "w") as f:
        json.dump({"build_latency_ms": latency}, f)
        
    return Zt

if __name__ == "__main__":
    Gt = np.random.randn(16, 16)
    As = np.random.randn(10)
    Ab = np.random.randn(10)
    Ft = np.random.randn(10)
    Cf = np.random.randn(1)
    Ct = np.random.randn(1)
    Et = np.random.randn(1)
    build_unified_state(Gt, As, Ab, Ft, Cf, Ct, Et)
