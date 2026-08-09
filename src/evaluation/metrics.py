from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def calculate_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)
    
def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
