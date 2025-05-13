# main.py
import argparse

def run_lstm(path_to_data, site_id):
    from algorithms.lstm_model import run_lstm
    run_lstm(path_to_data, site_id)

def main():
    parser = argparse.ArgumentParser(description="Traffic Volume Prediction System")
    parser.add_argument('--algo', type=str, choices=['lstm'], required=True, help='Algorithm to run')
    parser.add_argument('--data', type=str, default='Processed Data/traffic_model_ready.pkl', help='Path to dataset')
    parser.add_argument('--site', type=str, default='0970', help='Site ID to predict')

    args = parser.parse_args()

    if args.algo == 'lstm':
        run_lstm(args.data, args.site)

if __name__ == '__main__':
    main()
