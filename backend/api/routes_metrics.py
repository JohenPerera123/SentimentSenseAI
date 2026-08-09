from fastapi import APIRouter

router = APIRouter()

@router.get("/metrics")
def get_metrics():
    return {
        "imdb": {
            "logistic_regression": {"f1": 0.9072, "accuracy": 0.9072},
            "linear_svm": {"f1": 0.9052, "accuracy": 0.9052},
            "naive_bayes": {"f1": 0.8843, "accuracy": 0.8844},
            "distilbert": {"f1": 0.8536, "accuracy": 0.8536, "note": "Fair 5k - 5,000 training samples"},
            "distilbert_pilot": {"f1": 0.8080, "accuracy": 0.8080}
        },
        "twitter": {
            "logistic_regression": {"f1": 0.7732, "accuracy": 0.7732},
            "linear_svm": {"f1": 0.7556, "accuracy": 0.7556},
            "naive_bayes": {"f1": 0.7635, "accuracy": 0.7636},
            "distilbert": {"f1": 0.8034, "accuracy": 0.8034, "note": "Fair 5k - 5,000 training samples"},
            "distilbert_pilot": {"f1": 0.7118, "accuracy": 0.7120}
        }
    }
