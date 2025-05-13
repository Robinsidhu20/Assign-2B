import argparse
import importlib
from datetime import datetime
from algorithms.utils import load_processed_data
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description="Traffic Prediction System")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["lstm", "gru", "transformer"],  # Add future models here
                       help="Model type: lstm, gru, transformer, etc.")
    parser.add_argument("--site", type=str, required=True, 
                       help="SCATS site ID (e.g., 0970)")
    parser.add_argument("--start_date", type=str, required=True, 
                       help="Start date for prediction (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--predict_daily", action="store_true", 
                       help="Predict traffic for the next 24 hours")
    parser.add_argument("--target_time", type=str, 
                       help="Specific timestamp to predict (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--output", type=str, choices=["text", "plot"], default="text", 
                       help="Output format")
    return parser.parse_args()

def load_model(model_name):
    try:
        module = importlib.import_module(f"algorithms.{model_name}_model")
        model_class = getattr(module, f"{model_name.upper()}TrafficPredictor")
        return model_class
    except (ImportError, AttributeError):
        raise ValueError(f"Model '{model_name}' is not implemented or named incorrectly.")

def main():
    args = parse_arguments()
    
    # Load processed data for the specified site
    df = load_processed_data("Processed_Data/traffic_model_ready.pkl")
    site_data = df[df["Site_ID"] == args.site].sort_values("Timestamp")
    
    # Dynamically load the model class
    ModelClass = load_model(args.model)
    predictor = ModelClass(site_data)
    
    # Train/test split based on start date
    train, test = predictor.train_test_split(args.start_date)
    
    
    # Train model
    predictor.train_model()
    
    # Generate predictions
    if args.predict_daily:
        predictions = predictor.predict_daily(args.start_date)
    elif args.target_time:
        predictions = predictor.predict_specific_time(args.target_time)
    else:
        raise ValueError("Specify --predict_daily or --target_time")
    
    # Display results
    if args.output == "plot":
        predictions = predictions.reshape(-1, 1)
        scaler = predictor.scaler
        predictions_scaled = scaler.fit_transform(predictions)
        predictor.plot_predictions(test, predictions_scaled)
    else:
        print(f"Predicted Traffic Volume: {predictions}")

if __name__ == "__main__":
    main()