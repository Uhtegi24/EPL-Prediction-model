import os

try:
    from src.epl_prediction.data import load_data, preprocess_data
except ImportError:
    from src.data import load_data, preprocess_data

DATA_PATH = r"Data/epl_combined_2019_to_today.csv"
PROCESSED_PATH = r"Data/processed/matches_processed.csv"

def main():
    print("--- Starting Data Engineering Pipeline ---")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Input file not found at {DATA_PATH}")
        return
    
    raw_df = load_data(DATA_PATH)
    

    matches = preprocess_data(raw_df)
    

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    matches.to_csv(PROCESSED_PATH, index=False)
    
    print(f"Success! Transformed data saved to: {PROCESSED_PATH}")
    print(f"Total matches processed: {len(matches)}")
    print("Columns available for modeling:", list(matches.columns))

if __name__ == "__main__":
    main()