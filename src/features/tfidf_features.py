from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(max_features: int = 30000) -> TfidfVectorizer:
    """
    Returns a configured TF-IDF vectorizer.
    Using standard configs for sentiment analysis.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
