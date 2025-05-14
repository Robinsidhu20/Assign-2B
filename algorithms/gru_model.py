# algorithms/gru_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GRUTrafficPredictor:
    def __init__(self, data, sequence_length=96, epochs=30):
        """
        data: DataFrame with ['Timestamp','Site_ID','Volume']
        sequence_length: how many 15-min steps in your window
        epochs: training epochs
        """
        # 1) store & sort
        self.data = data.sort_values("Timestamp").reset_index(drop=True)
        self.sequence_length = sequence_length
        self.epochs = epochs

        # 2) prepare timestamps & raw volume
        self.timestamps = pd.to_datetime(self.data["Timestamp"])
        volumes = self.data["Volume"].values.reshape(-1,1)

        # 3) build cyclical time features
        hour = self.timestamps.dt.hour
        dow  = self.timestamps.dt.dayofweek
        features = np.hstack([
            volumes,
            np.sin(2*np.pi*hour/24).values.reshape(-1,1),
            np.cos(2*np.pi*hour/24).values.reshape(-1,1),
            np.sin(2*np.pi*dow/7).values.reshape(-1,1),
            np.cos(2*np.pi*dow/7).values.reshape(-1,1),
        ])

        # 4) scale everything
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.values_scaled = self.scaler.fit_transform(features)

        # placeholder
        self.model = None

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def train_test_split(self, split_date):
        """
        Returns:
          train_scaled: scaled data up to split_date
          test_scaled : last sequence_length points before split_date
        """
        idx = self.timestamps.searchsorted(pd.to_datetime(split_date))
        train_scaled = self.values_scaled[:idx]
        test_scaled  = self.values_scaled[idx - self.sequence_length : idx]
        return train_scaled, test_scaled

    def train_model(self):
        """Fit the GRU on historical values."""
        X, y = self._create_sequences(self.values_scaled)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        n_feat = X_train.shape[2]
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.sequence_length, n_feat)),
            tf.keras.layers.GRU(64, return_sequences=True, activation="tanh"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32, activation="tanh"),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=64,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )],
            verbose=2
        )

    def predict_daily(self, start_date):
        """Called by main.py when --predict_daily: returns next sequence_length forecasts."""
        idx = self.timestamps.searchsorted(pd.to_datetime(start_date))
        seed = self.values_scaled[idx - self.sequence_length : idx].copy()
        preds = []
        for _ in range(self.sequence_length):
            p = self.model.predict(seed[np.newaxis, ...])[0,0]
            preds.append(p)
            seed = np.vstack([seed[1:], [[p,0,0,0,0]]])
        arr = np.array(preds).reshape(-1,1)
        dummy = np.zeros((len(arr), self.values_scaled.shape[1]))
        dummy[:,0] = arr.flatten()
        return self.scaler.inverse_transform(dummy)[:,0]

    def predict_specific_time(self, target_time):
        """Called by main.py when --target_time: returns one-step forecast."""
        idx = self.timestamps.searchsorted(pd.to_datetime(target_time))
        seed = self.values_scaled[idx - self.sequence_length : idx].copy()
        p = self.model.predict(seed[np.newaxis, ...])[0,0]
        inv = self.scaler.inverse_transform([[p] + [0]*(self.values_scaled.shape[1]-1)])
        return inv[0,0]

    def plot_predictions(self, test_scaled, preds_scaled):
        """
        Called by main.py when --output plot.
        test_scaled: scaled window array from train_test_split
        preds_scaled: scaled array of your predictions
        """
        actual = self.scaler.inverse_transform(test_scaled)[:,0]
        padded = np.hstack([preds_scaled, np.zeros((len(preds_scaled), self.values_scaled.shape[1]-1))])
        preds  = self.scaler.inverse_transform(padded)[:,0]
        plt.figure(figsize=(10,5))
        plt.plot(actual, label="Actual")
        plt.plot(preds,  label="Predicted")
        plt.xlabel("15-min Steps")
        plt.ylabel("Traffic Volume")
        plt.legend()
        plt.show()
