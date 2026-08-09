import unittest
from src.preprocessing.text_cleaner import clean_text_basic
from src.preprocessing.tokenizer import NLPTokenizer
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

class TestPreprocessing(unittest.TestCase):
    def test_html_removal(self):
        text = "This movie is <b>great</b><br />."
        cleaned = clean_text_basic(text)
        self.assertEqual(cleaned, "This movie is great .")
        
    def test_url_removal(self):
        text = "Check this out https://example.com/movie it is good www.test.com"
        cleaned = clean_text_basic(text)
        self.assertEqual(cleaned, "Check this out it is good")
        
    def test_mention_handling(self):
        text = "@user1 I love this!"
        cleaned = clean_text_basic(text, is_twitter=True)
        self.assertEqual(cleaned, "I love this!")
        
    def test_whitespace_normalization(self):
        text = "This   is \n\n a test."
        cleaned = clean_text_basic(text)
        self.assertEqual(cleaned, "This is a test.")
        
    def test_twitter_hashtag(self):
        text = "#GreatMovie"
        cleaned = clean_text_basic(text, is_twitter=True)
        self.assertEqual(cleaned, "GreatMovie")
        
    def test_lowercasing_and_negation(self):
        tokenizer = NLPTokenizer()
        text = "I do NOT like this."
        tokens = tokenizer.tokenize(text)
        self.assertIn("not", tokens)
        self.assertIn("like", tokens)
        self.assertNotIn("i", tokens)
        self.assertNotIn("do", tokens)
        
    def test_empty_text(self):
        pipeline = PreprocessingPipeline()
        self.assertEqual(pipeline.process("   "), "")
        self.assertEqual(pipeline.process(None), "")
        
if __name__ == '__main__':
    unittest.main()
