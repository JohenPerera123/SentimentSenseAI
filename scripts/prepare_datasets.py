import os
import json
import pandas as pd
import sys

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.imdb_loader import load_imdb_dataset
from src.data.twitter_loader import load_twitter_dataset
from src.data.dataset_validator import validate_dataset
from src.data.data_utils import preliminary_clean, create_stratified_splits, save_splits

def main():
    print("Starting dataset preparation...")
    
    reports_dir = os.path.join("reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    imdb_path = "data/raw/imdb/IMDB Dataset.csv"
    twitter_path = "data/raw/twitter/twitter_dataset.csv"
    
    # 1. Load Datasets
    print(f"Loading IMDb from {imdb_path}...")
    df_imdb_raw = load_imdb_dataset(imdb_path)
    
    print(f"Loading Twitter from {twitter_path}...")
    df_twitter_raw = load_twitter_dataset(twitter_path, sample_size=50000)
    
    # 2. Validation Before Cleaning
    print("Validating datasets...")
    imdb_val = validate_dataset(df_imdb_raw, "IMDb")
    twitter_val = validate_dataset(df_twitter_raw, "Twitter (50k sample)")
    
    summary = {
        "datasets": [imdb_val, twitter_val]
    }
    
    with open(os.path.join(reports_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
        
    stats_df = pd.DataFrame([
        {
            "Dataset": val['dataset'],
            "Total_Rows": val['rows'],
            "Missing_Values": sum(val['missing_values'].values()),
            "Duplicate_Rows": val['duplicate_rows'],
            "Duplicate_Texts": val['duplicate_texts'],
            "Positive_Count": val['label_distribution'].get('positive', {}).get('count', 0),
            "Negative_Count": val['label_distribution'].get('negative', {}).get('count', 0)
        }
        for val in [imdb_val, twitter_val]
    ])
    stats_df.to_csv(os.path.join(reports_dir, "dataset_statistics.csv"), index=False)
    
    comparison_df = pd.DataFrame([
        {
            "Dataset": val['dataset'],
            "Total_Rows": val['rows'],
            "Classes": len(val['label_distribution']),
            "Missing_Values": sum(val['missing_values'].values()),
            "Duplicate_Rate_Pct": round(val['duplicate_rows'] / val['rows'] * 100, 2) if val['rows'] > 0 else 0,
            "Avg_Char_Length": val['text_statistics'].get('char_length', {}).get('mean', 0),
            "Median_Char_Length": val['text_statistics'].get('char_length', {}).get('median', 0),
            "Avg_Word_Count": val['text_statistics'].get('word_count', {}).get('mean', 0),
            "Median_Word_Count": val['text_statistics'].get('word_count', {}).get('median', 0)
        }
        for val in [imdb_val, twitter_val]
    ])
    comparison_df.to_csv(os.path.join(reports_dir, "dataset_comparison.csv"), index=False)
    
    # 3. Cleaning
    print("Performing preliminary cleaning...")
    df_imdb_clean = preliminary_clean(df_imdb_raw)
    df_twitter_clean = preliminary_clean(df_twitter_raw)
    
    # Save Twitter sample
    twitter_sample_path = os.path.join("data", "processed", "twitter_sample.csv")
    os.makedirs(os.path.dirname(twitter_sample_path), exist_ok=True)
    df_twitter_clean.to_csv(twitter_sample_path, index=False)
    
    # 4. Splits
    print("Creating splits...")
    imdb_train, imdb_val, imdb_test = create_stratified_splits(df_imdb_clean)
    save_splits(imdb_train, imdb_val, imdb_test, os.path.join("data", "processed", "imdb"))
    
    twitter_train, twitter_val, twitter_test = create_stratified_splits(df_twitter_clean)
    save_splits(twitter_train, twitter_val, twitter_test, os.path.join("data", "processed", "twitter"))
    
    print("Dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
