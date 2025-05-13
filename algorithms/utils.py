import pandas as pd
import pickle

def load_processed_data(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.endswith(".csv"):
        return pd.read_csv(path, parse_dates=["Timestamp"])
    else:
        raise ValueError("Unsupported file format")