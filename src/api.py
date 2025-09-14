"""
FastAPI application for model serving.
"""

import pickle
import logging
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import config
from schemas import PredictionRequest, PredictionResponse, HealthResponse, DiabetesDataPoint

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes using Random Forest model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


def load_model() -> None:
    """Load the trained model from disk."""
    global model
    try:
        model_path = Path(config.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions on diabetes data.
    
    Args:
        request: Prediction request containing data points
        
    Returns:
        Prediction response with predictions and probabilities
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert request data to DataFrame
        data_dicts = [dp.dict() for dp in request.data]
        df = pd.DataFrame(data_dicts)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]  # Probability of class 1
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist(),
            model_version=config.MODEL_VERSION
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_single", response_model=PredictionResponse)
async def predict_single(data_point: DiabetesDataPoint):
    """
    Make prediction on a single data point.
    
    Args:
        data_point: Single data point to predict
        
    Returns:
        Prediction response with predictions and probabilities
    """
    request = PredictionRequest(data=[data_point])
    return await predict(request)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=True
    )
