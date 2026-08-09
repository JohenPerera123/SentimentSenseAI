"""
Utility functions for dataset handling.
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def normalize_sentiment_labels(label: str) -> str:
    """
    Standardizes sentiment labels to lowercase 'positive' or 'negative'.
    """
    if pd.isna(label):
        return label
    label_str = str(label).strip().lower()
    if label_str in ['positive', 'pos', '1', '4']:
        return 'positive'
    elif label_str in ['negative', 'neg', '0']:
        return 'negative'
    return label_str

def create_stratified_splits(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'sentiment', 
                             random_state: int = 42) -> tuple:
    """
    Creates an 80/10/10 train/validation/test split.
    """
    X = df[[text_col]]
    y = df[label_col]
    
    # 80% train, 20% temp (for val and test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    
    # 10% val, 10% test (which is 50% of the 20% temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )
    
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, val_df, test_df
    
def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """
    Saves the split datasets to CSV files in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

def preliminary_clean(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'sentiment') -> pd.DataFrame:
    """
    Performs preliminary cleaning (removing nulls, empty texts, normalizing labels, handling basic whitespace).
    Does NOT do deep NLP preprocessing (e.g. stopword removal, lemmatization).
    """
    df_clean = df.copy()
    
    # Remove nulls
    df_clean = df_clean.dropna(subset=[text_col, label_col])
    
    # Normalize labels
    df_clean[label_col] = df_clean[label_col].apply(normalize_sentiment_labels)
    
    # Remove rows with invalid labels
    df_clean = df_clean[df_clean[label_col].isin(['positive', 'negative'])]
    
    # Normalize whitespace and drop empty texts
    df_clean[text_col] = df_clean[text_col].astype(str).str.strip()
    df_clean[text_col] = df_clean[text_col].str.replace(r'\s+', ' ', regex=True)
    df_clean = df_clean[df_clean[text_col].str.len() > 0]
    
    return df_clean
