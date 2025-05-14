import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GRUTrafficPredictor:
    def __init__(self, data, sequence_length=96, epochs=30):
        self.data = data
        self.sequence_length = sequence_length  # 24h × 4 intervals
        self.epochs = epochs
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
        # Preprocess
        self.timestamps = pd.to_datetime(data["Timestamp"])
        self.volume = data["Volume"].values.reshape(-1, 1)
        self.volume_scaled = self.scaler.fit_transform(self.volume)
        
    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def train_test_split(self, split_date):
        split_idx = self.timestamps.searchsorted(pd.to_datetime(split_date))
        train_data = self.volume_scaled[:split_idx]
        test_data = self.volume_scaled[split_idx - self.sequence_length:]
        return train_data, test_data
    
    def train_model(self):
        # build sequences
        X, y = self._create_sequences(self.volume_scaled)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # build GRU model
        self.model = tf.keras.Sequential([
            tf.keras.layers.GRU(
                50, return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(50),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse")
        
        # train
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
            verbose=2
        )
    
    def predict_daily(self, start_date):
        start_idx = self.timestamps.searchsorted(pd.to_datetime(start_date))
        seq = self.volume_scaled[start_idx - self.sequence_length : start_idx]
        preds = []
        for _ in range(self.sequence_length):
            p = self.model.predict(seq[np.newaxis, ...])[0,0]
            preds.append(p)
            seq = np.roll(seq, -1, axis=0)
            seq[-1] = p
        return self.scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    
    def predict_specific_time(self, target_time):
        idx = self.timestamps.searchsorted(pd.to_datetime(target_time))
        seq = self.volume_scaled[idx - self.sequence_length : idx]
        p = self.model.predict(seq[np.newaxis, ...])[0,0]
        return self.scaler.inverse_transform([[p]])[0][0]
    
    def plot_predictions(self, actual_scaled, preds_scaled):
        actual = self.scaler.inverse_transform(actual_scaled.reshape(-1,1)).flatten()
        preds  = self.scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
        plt.figure(figsize=(12,6))
        plt.plot(actual, label="Actual")
        plt.plot(preds,  label="GRU Predicted")
        plt.xlabel("15-min Intervals")
        plt.ylabel("Traffic Volume")
        plt.legend()
        plt.show()
