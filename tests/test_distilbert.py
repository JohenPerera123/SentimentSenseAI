import unittest
import os
import importlib.util

def check_transformers_installed():
    return importlib.util.find_spec('transformers') is not None and importlib.util.find_spec('torch') is not None

class TestDistilBert(unittest.TestCase):
    @unittest.skipIf(not check_transformers_installed(), "Transformers or PyTorch not installed")
    def test_distilbert_predictor(self):
        from src.models.distilbert_predictor import DistilBertPredictor
        
        # We can't guarantee a model is trained in CI, so we skip if the dir doesn't exist
        model_dir = os.path.join("models", "distilbert", "imdb")
        if not os.path.exists(model_dir):
            self.skipTest("DistilBERT model not found for testing.")
            
        predictor = DistilBertPredictor(model_dir)
        
        # Test basic inference
        result = predictor.predict("This movie is amazing.")
        self.assertIn("sentiment", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        
        self.assertTrue(0 <= result['sentiment'] <= 1)
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)
        self.assertEqual(len(result['probabilities']), 2)
        
        # Test empty
        empty_result = predictor.predict("")
        self.assertEqual(empty_result['sentiment'], 0)
        
        # Test long text
        long_result = predictor.predict("movie " * 500)
        self.assertIn("sentiment", long_result)

if __name__ == '__main__':
    unittest.main()
