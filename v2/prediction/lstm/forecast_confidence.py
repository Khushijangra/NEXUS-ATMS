import json
import numpy as np
import torch
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
out_dir = project_root / "v2" / "prediction" / "lstm"

def compute_confidence(y_true, y_pred, beta=1.0):
    # C_f = exp(-beta * ||x_hat - x||^2)
    squared_error = np.sum((y_pred - y_true) ** 2, axis=-1)
    return np.exp(-beta * squared_error)

def compute_ece(confidence, y_true, y_pred, bins=10):
    # Expected Calibration Error for regression is often computed via prediction intervals
    # but here we approximate based on the confidence metric provided
    return float(np.mean(np.abs(confidence - np.exp(-np.mean((y_pred - y_true)**2, axis=-1)))))

def compute_nll(y_true, y_pred, variance):
    # Negative Log Likelihood assuming Gaussian distribution
    return float(np.mean(0.5 * np.log(2 * np.pi * variance) + ((y_true - y_pred)**2) / (2 * variance)))

def compute_picp(y_true, y_pred, std_dev, z=1.96):
    # Prediction Interval Coverage Probability (95% CI)
    lower_bound = y_pred - z * std_dev
    upper_bound = y_pred + z * std_dev
    within_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)
    return float(np.mean(within_bounds))

def compute_mpiw(std_dev, z=1.96):
    # Mean Prediction Interval Width
    return float(np.mean(2 * z * std_dev))

def generate_confidence_metrics():
    dataset_dir = out_dir / "dataset"
    if not os.path.exists(dataset_dir / "test.npy"):
        return
        
    # In a full run, we would load the trained LSTM and run it on test data
    # Here we mock the output slightly for testing, or we load the real model if it exists
    model_path = out_dir / "lstm_best.pth"
    if not model_path.exists():
        print("LSTM model not found. Run lstm_predictor_wrapper.py first.")
        return
        
    from lstm_predictor_wrapper import LSTMPredictor, TrafficDataset
    from torch.utils.data import DataLoader
    
    test_data = np.load(dataset_dir / "test.npy")
    test_dataset = TrafficDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(input_dim=test_data.shape[1], forecast_dim=10).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
            
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    
    # Calculate global variance from residuals for standard deviation
    residuals = all_y_true - all_y_pred
    variance = np.var(residuals, axis=0) + 1e-6
    std_dev = np.sqrt(variance)
    
    beta = 1.0
    confidence = compute_confidence(all_y_true, all_y_pred, beta)
    
    metrics = {
        "Calibration Error (ECE)": compute_ece(confidence, all_y_true, all_y_pred),
        "Negative Log Likelihood (NLL)": compute_nll(all_y_true, all_y_pred, variance),
        "Prediction Interval Coverage (PICP)": compute_picp(all_y_true, all_y_pred, std_dev),
        "Mean Prediction Interval Width (MPIW)": compute_mpiw(std_dev)
    }
    
    with open(out_dir / "forecast_confidence.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Confidence metrics generated.")

if __name__ == "__main__":
    generate_confidence_metrics()
