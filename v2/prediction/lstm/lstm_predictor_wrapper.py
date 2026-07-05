import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

project_root = Path(__file__).resolve().parents[3]
out_dir = project_root / "v2" / "prediction" / "lstm"

class TrafficDataset(Dataset):
    def __init__(self, data, history=30, forecast=10):
        self.data = data
        self.history = history
        self.forecast = forecast
        
    def __len__(self):
        return len(self.data) - self.history - self.forecast + 1
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.history]
        y = self.data[idx + self.history : idx + self.history + self.forecast]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, forecast_dim, hidden1=128, hidden2=64, fc_dim=32, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc1 = nn.Linear(hidden2, fc_dim)
        # Output is (forecast_horizon * input_dim) to predict all features
        self.fc2 = nn.Linear(fc_dim, forecast_dim * input_dim)
        self.forecast_dim = forecast_dim
        self.input_dim = input_dim
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        # Take the last time step
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out.view(-1, self.forecast_dim, self.input_dim)

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def nrmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / (np.max(y_true) - np.min(y_true) + 1e-8)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def train_model():
    dataset_dir = out_dir / "dataset"
    if not os.path.exists(dataset_dir / "train.npy"):
        print("Dataset not found. Run lstm_dataset.py first.")
        return

    train_data = np.load(dataset_dir / "train.npy")
    val_data = np.load(dataset_dir / "val.npy")
    test_data = np.load(dataset_dir / "test.npy")
    
    history_len = 30
    forecast_len = 10
    batch_size = 64
    epochs = 100  # Will use early stopping
    
    train_dataset = TrafficDataset(train_data, history=history_len, forecast=forecast_len)
    val_dataset = TrafficDataset(val_data, history=history_len, forecast=forecast_len)
    test_dataset = TrafficDataset(test_data, history=history_len, forecast=forecast_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(input_dim=train_data.shape[1], forecast_dim=forecast_len).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                val_loss += criterion(y_pred, y).item()
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "lstm_best.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # Evaluate
    model.load_state_dict(torch.load(out_dir / "lstm_best.pth"))
    model.eval()
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
            
    all_y_true = np.concatenate(all_y_true).reshape(-1, train_data.shape[1])
    all_y_pred = np.concatenate(all_y_pred).reshape(-1, train_data.shape[1])
    
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(all_y_true, all_y_pred))),
        "MAE": float(mean_absolute_error(all_y_true, all_y_pred)),
        "MAPE": float(mape(all_y_true, all_y_pred)),
        "R2": float(r2_score(all_y_true, all_y_pred)),
        "SMAPE": float(smape(all_y_true, all_y_pred)),
        "NRMSE": float(nrmse(all_y_true, all_y_pred))
    }
    
    with open(out_dir / "forecast_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("LSTM Training complete. Metrics saved.")

if __name__ == "__main__":
    train_model()
