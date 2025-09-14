"""
Data schemas and validation models using Pydantic.
"""

from typing import Optional
from pydantic import BaseModel, Field, validator
import pandas as pd


class DiabetesDataPoint(BaseModel):
    """Schema for a single diabetes data point."""
    
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=300, description="Glucose level")
    blood_pressure: float = Field(..., ge=0, le=200, description="Blood pressure")
    skin_thickness: float = Field(..., ge=0, le=100, description="Skin thickness")
    insulin: float = Field(..., ge=0, le=1000, description="Insulin level")
    bmi: float = Field(..., ge=0, le=100, description="Body Mass Index")
    diabetes_pedigree_function: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    age: int = Field(..., ge=0, le=120, description="Age")
    
    @validator('glucose', 'blood_pressure', 'insulin', 'bmi')
    def validate_numeric_values(cls, v):
        """Validate that numeric values are reasonable."""
        if v < 0:
            raise ValueError('Value cannot be negative')
        return v
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            pd.DataFrame: lambda v: v.to_dict('records')
        }


class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    
    data: list[DiabetesDataPoint] = Field(..., description="List of data points to predict")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "data": [
                    {
                        "pregnancies": 6,
                        "glucose": 148,
                        "blood_pressure": 72,
                        "skin_thickness": 35,
                        "insulin": 0,
                        "bmi": 33.6,
                        "diabetes_pedigree_function": 0.627,
                        "age": 50
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    predictions: list[int] = Field(..., description="Predicted outcomes (0 or 1)")
    probabilities: list[float] = Field(..., description="Prediction probabilities")
    model_version: str = Field(..., description="Model version used for prediction")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "predictions": [1],
                "probabilities": [0.85],
                "model_version": "v1.0"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Service version")


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a DataFrame against the DiabetesDataPoint schema.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    # Rename columns to match schema
    column_mapping = {
        'Pregnancies': 'pregnancies',
        'Glucose': 'glucose', 
        'BloodPressure': 'blood_pressure',
        'SkinThickness': 'skin_thickness',
        'Insulin': 'insulin',
        'BMI': 'bmi',
        'DiabetesPedigreeFunction': 'diabetes_pedigree_function',
        'Age': 'age'
    }
    
    df_renamed = df.rename(columns=column_mapping)
    
    # Validate each row
    for idx, row in df_renamed.iterrows():
        try:
            DiabetesDataPoint(**row.to_dict())
        except Exception as e:
            raise ValueError(f"Validation failed for row {idx}: {str(e)}")
    
    return df_renamed
