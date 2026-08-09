import csv
import os
import random
from collections import Counter

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

def main():
    print("--- Phase 2: Dataset Loading and Validation ---")
    
    # 1. Process IMDb Dataset
    imdb_path = "data/raw/imdb/IMDB Dataset.csv"
    print(f"Loading IMDb dataset from {imdb_path}...")
    
    imdb_rows = []
    imdb_stats = Counter()
    missing_imdb = 0
    with open(imdb_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) != 2 or not row[0] or not row[1]:
                missing_imdb += 1
                continue
            imdb_rows.append((row[0], row[1]))
            imdb_stats[row[1]] += 1
            
    print(f"IMDb Shape: ({len(imdb_rows)}, 2)")
    print(f"IMDb Missing/Invalid Values: {missing_imdb}")
    
    imdb_unique = list(set(imdb_rows))
    duplicates = len(imdb_rows) - len(imdb_unique)
    print(f"IMDb Duplicates: {duplicates}")
    if duplicates > 0:
        print(f"Removing {duplicates} duplicates...")
        imdb_rows = imdb_unique
        # recompute stats
        imdb_stats = Counter([row[1] for row in imdb_rows])
        
    print(f"IMDb Class Distribution: {dict(imdb_stats)}")
    
    # 2. Process Twitter Dataset
    twitter_path = "data/raw/twitter/twitter_dataset.csv"
    print(f"\nLoading Twitter dataset from {twitter_path}...")
    
    twitter_rows = []
    twitter_original_stats = Counter()
    with open(twitter_path, 'r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        for row in reader:
            # target is at index 0, text is at index 5
            if len(row) < 6: continue
            target = row[0]
            text = row[5]
            twitter_original_stats[target] += 1
            if target in ['0', '4']:
                sentiment = 'negative' if target == '0' else 'positive'
                twitter_rows.append((text, sentiment))
                
    print(f"Twitter original class distribution (target): {dict(twitter_original_stats)}")
    print(f"Twitter Shape before sampling: ({len(twitter_rows)}, 2)")
    
    # Sample 50,000 (25,000 per class)
    sample_size = 50000
    pos_tweets = [r for r in twitter_rows if r[1] == 'positive']
    neg_tweets = [r for r in twitter_rows if r[1] == 'negative']
    
    if len(pos_tweets) > sample_size // 2:
        pos_tweets = random.sample(pos_tweets, sample_size // 2)
    if len(neg_tweets) > sample_size // 2:
        neg_tweets = random.sample(neg_tweets, sample_size // 2)
        
    twitter_sample = pos_tweets + neg_tweets
    random.shuffle(twitter_sample)
    
    print(f"Twitter Sample Shape: ({len(twitter_sample)}, 2)")
    print("Twitter Sample Missing Values: 0")
    
    twitter_sample_unique = list(set(twitter_sample))
    twitter_dups = len(twitter_sample) - len(twitter_sample_unique)
    print(f"Twitter Sample Duplicates (by text): {twitter_dups}")
    
    if twitter_dups > 0:
        print(f"Removing {twitter_dups} duplicates from Twitter sample...")
        twitter_sample = twitter_sample_unique
        
    twitter_stats = Counter([row[1] for row in twitter_sample])
    print(f"Twitter Sample Class Distribution: {dict(twitter_stats)}")
    
    # Save the processed Twitter sample
    processed_twitter_path = "data/processed/twitter_sample.csv"
    os.makedirs(os.path.dirname(processed_twitter_path), exist_ok=True)
    with open(processed_twitter_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'sentiment'])
        writer.writerows(twitter_sample)
    print(f"Saved Twitter sample to {processed_twitter_path}")
    
    # 3. Generate dataset statistics report
    os.makedirs('reports', exist_ok=True)
    with open('reports/dataset_statistics.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Total_Rows', 'Positive_Count', 'Negative_Count', 'Missing_Values', 'Duplicates_Removed'])
        writer.writerow(['IMDb', len(imdb_rows), imdb_stats.get('positive', 0), imdb_stats.get('negative', 0), missing_imdb, duplicates])
        writer.writerow(['Twitter (Sample)', len(twitter_sample), twitter_stats.get('positive', 0), twitter_stats.get('negative', 0), 0, twitter_dups])
    
    print("Saved dataset statistics to reports/dataset_statistics.csv")
    print("--- Phase 2 Complete ---")

if __name__ == "__main__":
    main()
