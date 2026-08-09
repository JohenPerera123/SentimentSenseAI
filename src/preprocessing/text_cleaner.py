import re

def remove_html_tags(text: str) -> str:
    """Removes HTML tags from text."""
    clean_r = re.compile('<.*?>')
    return re.sub(clean_r, ' ', text)

def remove_urls(text: str) -> str:
    """Removes HTTP/HTTPS URLs."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def clean_twitter_artifacts(text: str) -> str:
    """Removes @mentions and normalizes hashtags."""
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbol but keep the word (e.g., #AmazingMovie -> AmazingMovie)
    text = re.sub(r'#(\w+)', r'\1', text)
    return text

def normalize_whitespace(text: str) -> str:
    """Collapses multiple spaces/newlines into a single space."""
    return re.sub(r'\s+', ' ', text).strip()

def remove_repeated_characters(text: str) -> str:
    """Reduces characters repeated more than twice down to two (e.g., loooove -> loove)."""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def clean_text_basic(text: str, is_twitter: bool = False) -> str:
    """
    Applies the full sequence of basic cleaning steps.
    """
    text = str(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    
    if is_twitter:
        text = clean_twitter_artifacts(text)
        
    text = remove_repeated_characters(text)
    text = normalize_whitespace(text)
    return text
