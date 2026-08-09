import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import calculate_metrics, get_confusion_matrix

def evaluate_model(model, X_test, y_test, model_name: str, dataset_name: str):
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    cm = get_confusion_matrix(y_test, y_pred)
    classes = getattr(model, "classes_", ['negative', 'positive'])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
    
    reports_dir = os.path.join("reports", "figures")
    os.makedirs(reports_dir, exist_ok=True)
    filename = f"{dataset_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(os.path.join(reports_dir, filename))
    plt.close()
    
    metrics['model'] = model_name
    metrics['dataset'] = dataset_name
    return metrics, y_pred
