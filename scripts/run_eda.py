import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print('wordcloud not installed, skipping wordcloud generation')

os.makedirs('reports/figures', exist_ok=True)

# 1. Load data
imdb_df = pd.read_csv('data/processed/imdb/train.csv')
twitter_df = pd.read_csv('data/processed/twitter/train.csv')

# 3. Class Distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(data=imdb_df, x='sentiment', ax=ax[0])
ax[0].set_title('IMDb Class Distribution')
sns.countplot(data=twitter_df, x='sentiment', ax=ax[1])
ax[1].set_title('Twitter Class Distribution')
plt.savefig('reports/figures/imdb_class_distribution.png')
plt.savefig('reports/figures/twitter_class_distribution.png')
plt.close()

# 4. Text Length Distribution
imdb_df['char_length'] = imdb_df['text'].str.len()
twitter_df['char_length'] = twitter_df['text'].str.len()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(imdb_df['char_length'], bins=50, ax=ax[0])
ax[0].set_title('IMDb Text Length Distribution')
sns.histplot(twitter_df['char_length'], bins=50, ax=ax[1])
ax[1].set_title('Twitter Text Length Distribution')
plt.savefig('reports/figures/imdb_text_length_distribution.png')
plt.savefig('reports/figures/twitter_text_length_distribution.png')
plt.close()

# 5. Word Clouds
if WORDCLOUD_AVAILABLE:
    def generate_wordcloud(df, sentiment, title, filename):
        text = ' '.join(df[df['sentiment'] == sentiment]['text'].dropna().values)
        wc = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(title)
        plt.axis('off')
        plt.savefig(f'reports/figures/{filename}.png')
        plt.close()

    generate_wordcloud(imdb_df, 'positive', 'IMDb Positive', 'imdb_positive_wordcloud')
    generate_wordcloud(imdb_df, 'negative', 'IMDb Negative', 'imdb_negative_wordcloud')
    generate_wordcloud(twitter_df, 'positive', 'Twitter Positive', 'twitter_positive_wordcloud')
    generate_wordcloud(twitter_df, 'negative', 'Twitter Negative', 'twitter_negative_wordcloud')
else:
    print('Skipping wordcloud generation (dependency not met).')
