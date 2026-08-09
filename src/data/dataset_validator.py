import pandas as pd

def get_missing_values(df: pd.DataFrame) -> dict:
    """Returns a dictionary of missing value counts per column."""
    return df.isnull().sum().to_dict()

def get_duplicate_rows(df: pd.DataFrame, subset=None) -> int:
    """Returns the count of duplicate rows."""
    return df.duplicated(subset=subset).sum()

def get_label_distribution(df: pd.DataFrame, label_col: str = 'sentiment') -> dict:
    """Returns class distribution counts and percentages."""
    if label_col not in df.columns:
        return {}
    
    counts = df[label_col].value_counts()
    percentages = df[label_col].value_counts(normalize=True) * 100
    
    dist = {}
    for label, count in counts.items():
        dist[label] = {
            'count': int(count),
            'percentage': round(percentages[label], 2)
        }
    return dist

def get_text_statistics(df: pd.DataFrame, text_col: str = 'text') -> dict:
    """Returns statistics about text lengths (character and word counts)."""
    if text_col not in df.columns:
        return {}
        
    texts = df[text_col].dropna().astype(str)
    
    if len(texts) == 0:
        return {}
        
    char_lens = texts.str.len()
    word_counts = texts.str.split().str.len()
    
    return {
        'char_length': {
            'min': int(char_lens.min()),
            'max': int(char_lens.max()),
            'mean': round(float(char_lens.mean()), 2),
            'median': float(char_lens.median()),
            'std': round(float(char_lens.std()), 2)
        },
        'word_count': {
            'min': int(word_counts.min()),
            'max': int(word_counts.max()),
            'mean': round(float(word_counts.mean()), 2),
            'median': float(word_counts.median()),
            'std': round(float(word_counts.std()), 2)
        }
    }

def validate_dataset(df: pd.DataFrame, name: str, text_col: str = 'text', label_col: str = 'sentiment') -> dict:
    """Generates a complete validation report for a dataset."""
    return {
        'dataset': name,
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'missing_values': get_missing_values(df),
        'duplicate_rows': int(get_duplicate_rows(df)),
        'duplicate_texts': int(get_duplicate_rows(df, subset=[text_col])) if text_col in df.columns else 0,
        'label_distribution': get_label_distribution(df, label_col=label_col),
        'text_statistics': get_text_statistics(df, text_col=text_col)
    }
