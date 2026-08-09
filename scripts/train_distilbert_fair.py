import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation.metrics import calculate_metrics, get_confusion_matrix

# Fair Evaluation Configurations
MAX_LENGTH = 128
TRANSFORMER_TRAIN_SIZE = 5000
TRANSFORMER_VAL_SIZE = 1000
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=MAX_LENGTH)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    metrics = calculate_metrics(labels, predictions)
    return {
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1_score'],
        'precision': metrics['precision'],
        'recall': metrics['recall']
    }

def train_and_evaluate_distilbert_fair(dataset_name: str, train_df, val_df, test_df):
    print(f"--- Fair Fine-Tuning DistilBERT on {dataset_name} ---")
    
    # Stratified subsetting for Train and Validation, but NOT for Test.
    train_df = train_df.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(n=min(TRANSFORMER_TRAIN_SIZE // 2, len(x)), random_state=42)).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(n=min(TRANSFORMER_VAL_SIZE // 2, len(x)), random_state=42)).sample(frac=1, random_state=42).reset_index(drop=True)
    
    label_map = {"negative": 0, "positive": 1}
    train_df['sentiment'] = train_df['sentiment'].map(label_map).fillna(train_df['sentiment'])
    val_df['sentiment'] = val_df['sentiment'].map(label_map).fillna(val_df['sentiment'])
    test_df['sentiment'] = test_df['sentiment'].map(label_map).fillna(test_df['sentiment'])
    
    # Convert to HF Datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'sentiment']].rename(columns={'sentiment': 'label'}))
    val_dataset = Dataset.from_pandas(val_df[['text', 'sentiment']].rename(columns={'sentiment': 'label'}))
    test_dataset = Dataset.from_pandas(test_df[['text', 'sentiment']].rename(columns={'sentiment': 'label'}))
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_tokenized = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_tokenized = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    output_dir = f"./distilbert_fair_results_{dataset_name.lower()}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"./logs_fair_{dataset_name.lower()}",
        seed=42,
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        compute_metrics=compute_metrics,
    )
    
    print("Starting fair training run (this may take over an hour on CPU)...")
    trainer.train()
        
    print("Saving fair model...")
    save_dir = os.path.join("models", "distilbert_fair", dataset_name.lower())
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Predict on the full test set
    print("Evaluating on the official unmodified test set...")
    predictions = trainer.predict(test_tokenized)
    metrics = predictions.metrics
    
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_test = predictions.label_ids
    
    # Save confusion matrix
    cm = get_confusion_matrix(y_test, y_pred)
    classes = ['Negative', 'Positive']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: DistilBERT (Fair 5k) on {dataset_name}')
    reports_dir = os.path.join("reports", "figures")
    os.makedirs(reports_dir, exist_ok=True)
    plt.savefig(os.path.join(reports_dir, f"{dataset_name.lower()}_distilbert_fair_confusion_matrix.png"))
    plt.close()
    
    # Format metrics for final report
    final_metrics = {
        'dataset': dataset_name,
        'model': 'DistilBERT (Fair 5k)',
        'split': 'test',
        'accuracy': metrics.get('test_accuracy'),
        'precision': metrics.get('test_precision'),
        'recall': metrics.get('test_recall'),
        'f1_score': metrics.get('test_f1')
    }
    
    return pd.DataFrame([final_metrics])

def main():
    os.makedirs(os.path.join("models", "distilbert_fair"), exist_ok=True)
    
    # Load actual data
    imdb_train = pd.read_csv("data/processed/imdb/train.csv")
    imdb_val = pd.read_csv("data/processed/imdb/validation.csv")
    imdb_test = pd.read_csv("data/processed/imdb/test.csv")
    
    twitter_train = pd.read_csv("data/processed/twitter/train.csv")
    twitter_val = pd.read_csv("data/processed/twitter/validation.csv")
    twitter_test = pd.read_csv("data/processed/twitter/test.csv")
    
    imdb_res = train_and_evaluate_distilbert_fair("IMDb", imdb_train, imdb_val, imdb_test)
    twitter_res = train_and_evaluate_distilbert_fair("Twitter", twitter_train, twitter_val, twitter_test)
    
    all_results_list = [res for res in [imdb_res, twitter_res] if res is not None]
    
    if len(all_results_list) > 0:
        all_results = pd.concat(all_results_list, ignore_index=True)
        # Append to main results
        main_results = pd.read_csv("reports/model_results.csv")
        updated_results = pd.concat([main_results, all_results], ignore_index=True)
        updated_results.to_csv("reports/model_results.csv", index=False)
        
    print("DistilBERT Fair Evaluation complete!")

if __name__ == "__main__":
    main()
