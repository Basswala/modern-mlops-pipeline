"""
Tests for data schemas and validation.
"""

import pytest
import pandas as pd
from pydantic import ValidationError

from src.schemas import DiabetesDataPoint, validate_dataframe


class TestDiabetesDataPoint:
    """Test DiabetesDataPoint schema validation."""
    
    def test_valid_data_point(self):
        """Test valid data point creation."""
        data = DiabetesDataPoint(
            pregnancies=6,
            glucose=148,
            blood_pressure=72,
            skin_thickness=35,
            insulin=0,
            bmi=33.6,
            diabetes_pedigree_function=0.627,
            age=50
        )
        assert data.pregnancies == 6
        assert data.glucose == 148
    
    def test_invalid_negative_values(self):
        """Test validation of negative values."""
        with pytest.raises(ValidationError):
            DiabetesDataPoint(
                pregnancies=-1,  # Invalid
                glucose=148,
                blood_pressure=72,
                skin_thickness=35,
                insulin=0,
                bmi=33.6,
                diabetes_pedigree_function=0.627,
                age=50
            )
    
    def test_boundary_values(self):
        """Test boundary value validation."""
        # Test maximum values
        data = DiabetesDataPoint(
            pregnancies=20,  # Max value
            glucose=300,     # Max value
            blood_pressure=200,  # Max value
            skin_thickness=100,  # Max value
            insulin=1000,    # Max value
            bmi=100,         # Max value
            diabetes_pedigree_function=3,  # Max value
            age=120          # Max value
        )
        assert data.pregnancies == 20
    
    def test_exceeding_boundary_values(self):
        """Test validation when exceeding maximum values."""
        with pytest.raises(ValidationError):
            DiabetesDataPoint(
                pregnancies=21,  # Exceeds max
                glucose=148,
                blood_pressure=72,
                skin_thickness=35,
                insulin=0,
                bmi=33.6,
                diabetes_pedigree_function=0.627,
                age=50
            )


class TestDataFrameValidation:
    """Test DataFrame validation functions."""
    
    def test_valid_dataframe_validation(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({
            'Pregnancies': [6, 1],
            'Glucose': [148, 85],
            'BloodPressure': [72, 66],
            'SkinThickness': [35, 29],
            'Insulin': [0, 0],
            'BMI': [33.6, 26.6],
            'DiabetesPedigreeFunction': [0.627, 0.351],
            'Age': [50, 31],
            'Outcome': [1, 0]
        })
        
        validated_df = validate_dataframe(df)
        assert validated_df.shape == (2, 9)
        assert 'pregnancies' in validated_df.columns
    
    def test_invalid_dataframe_validation(self):
        """Test validation of invalid DataFrame."""
        df = pd.DataFrame({
            'Pregnancies': [-1, 1],  # Invalid negative value
            'Glucose': [148, 85],
            'BloodPressure': [72, 66],
            'SkinThickness': [35, 29],
            'Insulin': [0, 0],
            'BMI': [33.6, 26.6],
            'DiabetesPedigreeFunction': [0.627, 0.351],
            'Age': [50, 31],
            'Outcome': [1, 0]
        })
        
        with pytest.raises(ValueError, match="Validation failed"):
            validate_dataframe(df)
    
    def test_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({
            'Pregnancies': [6, 1],
            'Glucose': [148, 85],
            # Missing other required columns
        })
        
        with pytest.raises(KeyError):
            validate_dataframe(df)
