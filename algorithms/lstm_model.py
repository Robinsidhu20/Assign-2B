# algorithms/lstm_model.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = self.fc(out[:, -1, :])     # Only take the last time step
        return out

def prepare_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def run_lstm(path_to_data, site_id='0970', epochs=10):
    # Load and filter data
    df = pd.read_pickle(path_to_data)
    df = df[df['Site_ID'] == site_id].copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')

    volumes = df['Volume'].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    volumes_scaled = scaler.fit_transform(volumes)

    # Sequence length
    seq_length = 7
    X, y = prepare_sequences(volumes_scaled, seq_length)

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Prediction
    with torch.no_grad():
        last_seq = torch.tensor(volumes_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)
        prediction = model(last_seq).item()
        prediction_inverse = scaler.inverse_transform([[prediction]])[0][0]
        print(f"\nNext predicted traffic volume: {prediction_inverse:.2f}")
