"""
Configuration management for the ML pipeline.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration class."""
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", 
        "https://dagshub.com/basswala/machinelearningpipeline.mlflow"
    )
    MLFLOW_TRACKING_USERNAME: Optional[str] = os.getenv("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD: Optional[str] = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("MODEL_NAME", "diabetes_classifier")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "latest")
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    
    # Data Configuration
    RAW_DATA_PATH: str = os.getenv("RAW_DATA_PATH", "data/raw/data.csv")
    PROCESSED_DATA_PATH: str = os.getenv("PROCESSED_DATA_PATH", "data/processed/data.csv")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/model.pkl")
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration values."""
        required_vars = [
            "MLFLOW_TRACKING_USERNAME",
            "MLFLOW_TRACKING_PASSWORD"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )


# Global config instance
config = Config()
