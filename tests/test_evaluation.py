import unittest
from src.evaluation.metrics import calculate_metrics

class TestEvaluation(unittest.TestCase):
    def test_calculate_metrics(self):
        y_true = ["positive", "positive", "negative", "negative"]
        y_pred = ["positive", "negative", "negative", "negative"]
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIn("f1_score", metrics)
        
if __name__ == '__main__':
    unittest.main()
