import unittest
import numpy as np
from src.models.train_naive_bayes import get_naive_bayes_model

class TestModels(unittest.TestCase):
    def test_naive_bayes(self):
        model = get_naive_bayes_model()
        # Create dummy data
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y = np.array(["positive", "negative", "positive", "negative"])
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), 4)

if __name__ == '__main__':
    unittest.main()
