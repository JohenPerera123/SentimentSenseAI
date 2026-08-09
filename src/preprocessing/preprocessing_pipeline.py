from .text_cleaner import clean_text_basic
from .tokenizer import NLPTokenizer

class PreprocessingPipeline:
    def __init__(self, is_twitter: bool = False):
        self.is_twitter = is_twitter
        self.tokenizer = NLPTokenizer()
        
    def process(self, text: str) -> str:
        """
        Full pipeline: Clean -> Tokenize -> Lemmatize -> Join back to string (for TF-IDF).
        """
        if text is None or str(text).strip() == "":
            return ""
            
        cleaned = clean_text_basic(text, self.is_twitter)
        tokens = self.tokenizer.tokenize(cleaned)
        
        return " ".join(tokens)
