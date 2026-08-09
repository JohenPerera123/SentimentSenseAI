import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class NLPTokenizer:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback if download failed
            self.stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                               "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
                               "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                               "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                               "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
                               "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
                               "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
                               "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
                               "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
                               "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
                               "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very",
                               "s", "t", "can", "will", "just", "don", "should", "now"}

        # Sentiment-critical words we must preserve (negation mostly)
        self.negation_words = {"not", "no", "never", "none", "cannot", "isn't", "aren't", "wasn't", "weren't", 
                               "haven't", "hasn't", "hadn't", "won't", "wouldn't", "don't", "doesn't", "didn't", 
                               "can't", "couldn't", "shouldn't", "mightn't", "mustn't", "nor"}
        
        # Remove negation words from stop words if they are present
        self.stop_words = self.stop_words - self.negation_words
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None

    def tokenize(self, text: str) -> list:
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize by finding words and preserving standard negations (like don't)
        # We'll use a regex that captures words with optional internal apostrophes
        tokens = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text)
        
        cleaned_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                if self.lemmatizer:
                    try:
                        token = self.lemmatizer.lemmatize(token)
                    except:
                        pass # fallback if lemmatization fails
                cleaned_tokens.append(token)
                
        return cleaned_tokens
