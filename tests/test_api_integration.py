import unittest
from fastapi.testclient import TestClient
from backend.main import app

class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_logistic_regression_positive(self):
        response = self.client.post("/api/predict", json={
            "text": "This movie was absolutely amazing! I loved every minute of it.",
            "model": "logistic_regression",
            "domain": "imdb"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sentiment"], "positive")
        self.assertIn("confidence", data)
        self.assertIsInstance(data["confidence"], float)

    def test_predict_logistic_regression_negative(self):
        response = self.client.post("/api/predict", json={
            "text": "This movie was terrible. I hated it and it was a complete waste of time.",
            "model": "logistic_regression",
            "domain": "imdb"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["sentiment"], "negative")
        self.assertIn("confidence", data)
        self.assertIsInstance(data["confidence"], float)

    def test_predict_linear_svm(self):
        response = self.client.post("/api/predict", json={
            "text": "Not a great experience.",
            "model": "linear_svm",
            "domain": "imdb"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["sentiment"], ["positive", "negative"])
        self.assertIn("confidence", data)

    def test_predict_distilbert(self):
        response = self.client.post("/api/predict", json={
            "text": "Just okay I guess.",
            "model": "distilbert",
            "domain": "imdb"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["sentiment"], ["positive", "negative"])
        self.assertIn("confidence", data)

    def test_compare_endpoint(self):
        response = self.client.post("/api/compare", json={
            "text": "This movie was absolutely amazing! I loved every minute of it.",
            "domain": "imdb"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predictions", data)
        self.assertGreaterEqual(len(data["predictions"]), 1)
        for p in data["predictions"]:
            self.assertIn("model", p)
            self.assertIn("sentiment", p)
            self.assertIn("confidence", p)

if __name__ == "__main__":
    unittest.main()
