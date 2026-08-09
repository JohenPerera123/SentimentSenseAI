from pydantic import BaseModel, constr

class PredictRequest(BaseModel):
    text: constr(min_length=1, max_length=5000)
    model: str
    domain: str

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    model: str
    domain: str

class CompareRequest(BaseModel):
    text: constr(min_length=1, max_length=5000)
    domain: str

class ModelPrediction(BaseModel):
    model: str
    sentiment: str
    confidence: float

class CompareResponse(BaseModel):
    predictions: list[ModelPrediction]

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool

class ModelsResponse(BaseModel):
    models: list[str]
    domains: list[str]
