import pandas as pd
import os

def load_imdb_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the raw IMDb dataset.
    
    Args:
        filepath: Path to the IMDb CSV file.
    
    Returns:
        DataFrame containing 'text' and 'sentiment' columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"IMDb dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Standardize column name
    if 'review' in df.columns:
        df = df.rename(columns={'review': 'text'})
        
    return df
