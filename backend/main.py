from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import routes_predict, routes_metrics
from backend.schemas.schemas import HealthResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SentimentScope API",
    description="Multi-Model Sentiment Analysis API",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(routes_predict.router, prefix="/api", tags=["Predict"])
app.include_router(routes_metrics.router, prefix="/api", tags=["Metrics"])

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="ok",
        models_loaded=True
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
