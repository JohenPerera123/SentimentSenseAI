import pandas as pd
import os
from .data_utils import normalize_sentiment_labels

def load_twitter_dataset(filepath: str, sample_size: int = None, random_state: int = 42) -> pd.DataFrame:
    """
    Loads the raw Twitter dataset (Sentiment140) and applies reproducible sampling.
    
    Args:
        filepath: Path to the Twitter CSV file.
        sample_size: The number of tweets to sample. If None, returns all matching tweets.
        random_state: Seed for reproducible sampling.
        
    Returns:
        DataFrame containing 'text' and 'sentiment' columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Twitter dataset not found at {filepath}")
        
    col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(filepath, encoding='ISO-8859-1', names=col_names)
    
    # Filter out neutral sentiment if present, only keep 0 and 4
    df = df[df['target'].isin([0, 4])].copy()
    
    # Map to standardized schema
    df['sentiment'] = df['target'].apply(normalize_sentiment_labels)
    df = df[['text', 'sentiment']]
    
    if sample_size and len(df) > sample_size:
        # Stratified sampling
        df = df.groupby('sentiment', group_keys=False).apply(
            lambda x: x.sample(n=sample_size // 2, random_state=random_state)
        ).sample(frac=1, random_state=random_state).reset_index(drop=True)
        
    return df
