from fastapi import APIRouter, HTTPException
from backend.schemas.schemas import PredictRequest, PredictResponse, CompareRequest, CompareResponse
import sys
import os

# Add parent path to reach src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.model_service import ModelService

router = APIRouter()
model_service = ModelService()

@router.get("/models")
def get_models():
    return {
        "models": ["logistic_regression", "linear_svm", "distilbert"],
        "domains": ["imdb", "twitter"]
    }

@router.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    try:
        res = model_service.predict(request.text, request.model, request.domain)
        sentiment_str = "positive" if res['sentiment'] == 1 else "negative"
        return PredictResponse(
            text=request.text,
            sentiment=sentiment_str,
            confidence=round(res['confidence'], 4),
            model=request.model,
            domain=request.domain
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/compare", response_model=CompareResponse)
def compare_models(request: CompareRequest):
    try:
        res = model_service.compare(request.text, request.domain)
        return CompareResponse(predictions=res["predictions"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
