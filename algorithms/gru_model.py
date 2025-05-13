import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

class GRUTrafficPredictor:
    def __init__(self, data, sequence_length=96, epochs=30):
        self.data = data
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
        # Preprocess data (same as LSTM)
        self.timestamps = pd.to_datetime(data["Timestamp"])
        self.volume = data["Volume"].values.reshape(-1, 1)
        self.volume_scaled = self.scaler.fit_transform(self.volume)
    
    def _create_sequences(self, data):
        # Same as LSTM
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train_test_split(self, split_date):
        # Same as LSTM
        split_idx = self.timestamps.searchsorted(pd.to_datetime(split_date))
        train_data = self.volume_scaled[:split_idx]
        test_data = self.volume_scaled[split_idx - self.sequence_length:]
        return train_data, test_data
    
    def train_model(self):
        X, y = self._create_sequences(self.volume_scaled)
        # GRU-specific architecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.GRU(50, return_sequences=True, 
                              input_shape=(X.shape[1], X.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(50),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(X, y, epochs=self.epochs, validation_split=0.2)
    
    def predict_daily(self, start_date):
        # Same as LSTM
        start_idx = self.timestamps.searchsorted(pd.to_datetime(start_date))
        input_sequence = self.volume_scaled[start_idx - self.sequence_length : start_idx]
        predictions = []
        for _ in range(96):  # Predict next 24 hours (96 intervals)
            pred = self.model.predict(input_sequence[np.newaxis, ...])
            predictions.append(pred[0, 0])
            input_sequence = np.roll(input_sequence, -1, axis=0)
            input_sequence[-1] = pred
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    def predict_specific_time(self, target_time):
        # Same as LSTM
        target_idx = self.timestamps.searchsorted(pd.to_datetime(target_time))
        input_sequence = self.volume_scaled[target_idx - self.sequence_length : target_idx]
        prediction = self.model.predict(input_sequence[np.newaxis, ...])
        return self.scaler.inverse_transform(prediction)[0][0]
    
    def plot_predictions(self, test_data, predictions):
        # Same as LSTM
        plt.figure(figsize=(12, 6))
        plt.plot(test_data, label="Actual Traffic")
        plt.plot(predictions, label="Predicted Traffic")
        plt.xlabel("Time Intervals (15-min)")
        plt.ylabel("Traffic Volume")
        plt.legend()
        plt.show()