import unittest
from src.features.tfidf_features import get_tfidf_vectorizer

class TestTFIDF(unittest.TestCase):
    def test_tfidf_initialization(self):
        vec = get_tfidf_vectorizer()
        self.assertEqual(vec.max_features, 30000)
        self.assertEqual(vec.ngram_range, (1, 2))
        
    def test_tfidf_fit_transform(self):
        vec = get_tfidf_vectorizer(max_features=10)
        vec.set_params(min_df=1) # Prevent dropping terms in small mock dataset
        train_texts = ["not good", "very bad movie", "great film", "good movie"]
        test_texts = ["good movie"]
        
        # Fit ONLY on train
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)
        
        self.assertEqual(X_train.shape[0], 4)
        self.assertEqual(X_test.shape[0], 1)
        self.assertTrue(X_train.shape[1] > 0)
        
if __name__ == '__main__':
    unittest.main()
