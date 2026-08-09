import unittest
import pandas as pd
from src.data.imdb_loader import load_imdb_dataset
from src.data.twitter_loader import load_twitter_dataset
from src.data.data_utils import preliminary_clean, normalize_sentiment_labels

class TestDataValidation(unittest.TestCase):
    def test_normalize_sentiment_labels(self):
        self.assertEqual(normalize_sentiment_labels("Positive"), "positive")
        self.assertEqual(normalize_sentiment_labels("4"), "positive")
        self.assertEqual(normalize_sentiment_labels("0"), "negative")
        self.assertEqual(normalize_sentiment_labels("Negative"), "negative")
        self.assertEqual(normalize_sentiment_labels("unknown"), "unknown")

    def test_preliminary_clean(self):
        df = pd.DataFrame({
            "text": ["  hello world  ", "   ", None, "valid text"],
            "sentiment": ["positive", "negative", "positive", "invalid_label"]
        })
        
        cleaned = preliminary_clean(df)
        
        # Should only keep row 0 (row 1 is empty text, row 2 is null, row 3 has invalid label)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["text"], "hello world")
        self.assertEqual(cleaned.iloc[0]["sentiment"], "positive")

if __name__ == '__main__':
    unittest.main()
