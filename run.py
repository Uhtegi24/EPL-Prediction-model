import os
import pandas as pd

# Try importing from the package structure, fallback to direct src if moved
try:
    from src.epl_prediction.data import load_data, preprocess_data
except ImportError:
    from src.data import load_data, preprocess_data

# Configuration
DATA_PATH = r"Data/epl_combined_2019_to_today.csv"
PROCESSED_PATH = r"Data/processed/matches_processed.csv"

def main():
    print("--- Starting Data Engineering Pipeline ---")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Input file not found at {DATA_PATH}")
        return
    
    raw_df = load_data(DATA_PATH)
    
    # 2. Transform Data (Clean + Feature Engineering)
    # This runs the logic in data.py including rolling averages
    matches = preprocess_data(raw_df)
    
    # 3. Save Processed Data
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    matches.to_csv(PROCESSED_PATH, index=False)
    
    print(f"Success! Transformed data saved to: {PROCESSED_PATH}")
    print(f"Total matches processed: {len(matches)}")
    print("Columns available for modeling:", list(matches.columns))

    # 2. Define Features
    # We ONLY use the rolling features (history), not the current match stats
    features = [
        "home_roll_shots", "home_roll_xg", "home_roll_sot", "home_roll_points", "home_rest_days",
        "away_roll_shots", "away_roll_xg", "away_roll_sot", "away_roll_points", "away_rest_days"
    ]

if __name__ == "__main__":
    main()