import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.features.tfidf_features import get_tfidf_vectorizer
from src.models.train_naive_bayes import get_naive_bayes_model
from src.models.train_logistic_regression import get_logistic_regression_model
from src.models.train_svm import get_linear_svm_model
from src.models.model_utils import save_model
from src.evaluation.evaluator import evaluate_model
from sklearn.dummy import DummyClassifier

def train_and_evaluate(dataset_name: str, train_df, val_df, test_df, is_twitter: bool):
    print(f"--- Processing {dataset_name} ---")
    
    pipeline = PreprocessingPipeline(is_twitter=is_twitter)
    
    print("Preprocessing text...")
    X_train_text = train_df['text'].apply(pipeline.process)
    X_val_text = val_df['text'].apply(pipeline.process)
    X_test_text = test_df['text'].apply(pipeline.process)
    
    y_train = train_df['sentiment']
    y_val = val_df['sentiment']
    y_test = test_df['sentiment']
    
    print("Fitting TF-IDF on training data only...")
    vectorizer = get_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    
    print("Transforming validation and test data...")
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)
    
    models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": get_naive_bayes_model(),
        "Logistic Regression": get_logistic_regression_model(),
        "Linear SVM": get_linear_svm_model()
    }
    
    results = []
    
    best_model_name = None
    best_f1 = -1
    best_model = None
    best_y_pred = None
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        
        print(f"Evaluating {model_name} on validation set...")
        metrics, y_pred = evaluate_model(model, X_val, y_val, model_name, dataset_name)
        
        # Save model
        save_model(model, vectorizer, os.path.join("models", dataset_name.lower(), model_name.lower().replace(" ", "_")))
        
        results.append(metrics)
        
        if metrics['f1_score'] > best_f1 and model_name != "Baseline":
            best_f1 = metrics['f1_score']
            best_model_name = model_name
            best_model = model
            best_y_pred = y_pred

    # Error analysis for the best model
    print(f"Best model for {dataset_name} is {best_model_name}. Generating error analysis...")
    errors_df = pd.DataFrame({
        'text': val_df['text'],
        'actual_label': y_val,
        'predicted_label': best_y_pred,
        'model': best_model_name
    })
    
    errors = errors_df[errors_df['actual_label'] != errors_df['predicted_label']]
    errors_sample = errors.sample(min(200, len(errors)), random_state=42)
    errors_sample.to_csv(os.path.join("reports", f"{dataset_name.lower()}_traditional_model_errors.csv"), index=False)
    
    # Feature importance for Logistic Regression if we trained it
    if "Logistic Regression" in models:
        lr_model = models["Logistic Regression"]
        feature_names = vectorizer.get_feature_names_out()
        coefs = lr_model.coef_[0]
        
        top_pos_idx = np.argsort(coefs)[-20:]
        top_neg_idx = np.argsort(coefs)[:20]
        
        feature_data = []
        for idx in top_pos_idx:
            feature_data.append({"dataset": dataset_name, "model": "Logistic Regression", "feature": feature_names[idx], "weight": coefs[idx], "sentiment_direction": "positive"})
        for idx in top_neg_idx:
            feature_data.append({"dataset": dataset_name, "model": "Logistic Regression", "feature": feature_names[idx], "weight": coefs[idx], "sentiment_direction": "negative"})
            
        pd.DataFrame(feature_data).to_csv(os.path.join("reports", f"{dataset_name.lower()}_tfidf_top_features.csv"), index=False)

    return pd.DataFrame(results)

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    
    all_results = []
    
    # IMDb
    imdb_train = pd.read_csv("data/processed/imdb/train.csv")
    imdb_val = pd.read_csv("data/processed/imdb/validation.csv")
    imdb_test = pd.read_csv("data/processed/imdb/test.csv")
    
    imdb_results = train_and_evaluate("IMDb", imdb_train, imdb_val, imdb_test, is_twitter=False)
    all_results.append(imdb_results)
    
    # Twitter
    twitter_train = pd.read_csv("data/processed/twitter/train.csv")
    twitter_val = pd.read_csv("data/processed/twitter/validation.csv")
    twitter_test = pd.read_csv("data/processed/twitter/test.csv")
    
    twitter_results = train_and_evaluate("Twitter", twitter_train, twitter_val, twitter_test, is_twitter=True)
    all_results.append(twitter_results)
    
    # Save combined results
    final_results = pd.concat(all_results, ignore_index=True)
    final_results = final_results[['dataset', 'model', 'accuracy', 'precision', 'recall', 'f1_score']]
    final_results.to_csv("reports/model_results.csv", index=False)
    
    # Generate model comparison chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=final_results[final_results['dataset'] == 'IMDb'], x='model', y='f1_score')
    plt.title('IMDb Traditional Model Comparison (F1 Score)')
    plt.savefig('reports/figures/imdb_traditional_model_comparison.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=final_results[final_results['dataset'] == 'Twitter'], x='model', y='f1_score')
    plt.title('Twitter Traditional Model Comparison (F1 Score)')
    plt.savefig('reports/figures/twitter_traditional_model_comparison.png')
    plt.close()
    
    print("Training complete! Results saved in reports/model_results.csv")

if __name__ == "__main__":
    main()
