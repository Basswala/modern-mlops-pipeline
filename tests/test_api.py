"""
Tests for the FastAPI application.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api import app
from src.schemas import DiabetesDataPoint, PredictionRequest

client = TestClient(app)


@pytest.fixture
def sample_data_point():
    """Sample data point for testing."""
    return DiabetesDataPoint(
        pregnancies=6,
        glucose=148,
        blood_pressure=72,
        skin_thickness=35,
        insulin=0,
        bmi=33.6,
        diabetes_pedigree_function=0.627,
        age=50
    )


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.predict.return_value = [1]
    model.predict_proba.return_value = [[0.2, 0.8]]
    return model


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


@patch('src.api.model')
def test_predict_single(mock_model, sample_data_point):
    """Test single prediction endpoint."""
    # Set up mock
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    
    response = client.post("/predict_single", json=sample_data_point.dict())
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert "model_version" in data


@patch('src.api.model')
def test_predict_batch(mock_model, sample_data_point):
    """Test batch prediction endpoint."""
    # Set up mock
    mock_model.predict.return_value = [1, 0]
    mock_model.predict_proba.return_value = [[0.2, 0.8], [0.9, 0.1]]
    
    request_data = PredictionRequest(data=[sample_data_point, sample_data_point])
    
    response = client.post("/predict", json=request_data.dict())
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert len(data["probabilities"]) == 2


@patch('src.api.model', None)
def test_predict_model_not_loaded():
    """Test prediction when model is not loaded."""
    sample_data = DiabetesDataPoint(
        pregnancies=6,
        glucose=148,
        blood_pressure=72,
        skin_thickness=35,
        insulin=0,
        bmi=33.6,
        diabetes_pedigree_function=0.627,
        age=50
    )
    
    response = client.post("/predict_single", json=sample_data.dict())
    
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_invalid_data_validation():
    """Test data validation with invalid input."""
    invalid_data = {
        "pregnancies": -1,  # Invalid: negative value
        "glucose": 148,
        "blood_pressure": 72,
        "skin_thickness": 35,
        "insulin": 0,
        "bmi": 33.6,
        "diabetes_pedigree_function": 0.627,
        "age": 50
    }
    
    response = client.post("/predict_single", json=invalid_data)
    
    assert response.status_code == 422  # Validation error
