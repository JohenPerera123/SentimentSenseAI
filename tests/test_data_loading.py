import unittest
from src.data.imdb_loader import load_imdb_dataset
from src.data.twitter_loader import load_twitter_dataset

class TestDataLoading(unittest.TestCase):
    def test_imdb_loader_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_imdb_dataset("nonexistent_path.csv")

    def test_twitter_loader_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_twitter_dataset("nonexistent_path.csv")

if __name__ == '__main__':
    unittest.main()
