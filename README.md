# 🤖 Modern MLOps Pipeline - Diabetes Prediction

[![CI/CD](https://github.com/basswala/diabetes-ml-pipeline/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/basswala/diabetes-ml-pipeline/actions)
[![codecov](https://codecov.io/gh/basswala/diabetes-ml-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/basswala/diabetes-ml-pipeline)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, production-ready machine learning pipeline for diabetes prediction using Random Forest Classifier. This project demonstrates best practices in MLOps including containerization, CI/CD, testing, monitoring, and API serving.

## 🚀 Features

- **🔄 Reproducible Pipeline**: DVC-based data versioning and pipeline orchestration
- **📊 Experiment Tracking**: MLflow integration for metrics and model tracking
- **🐳 Containerized**: Docker support for consistent deployments
- **🔧 API Serving**: FastAPI-based model serving with automatic validation
- **✅ Comprehensive Testing**: Unit tests, integration tests, and API tests
- **🔄 CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **📝 Data Validation**: Pydantic-based data schema validation
- **🔒 Security**: Environment-based configuration management
- **📈 Monitoring**: Health checks and observability features

## 🛠️ Technology Stack

### Core ML
- **Python +**: Core programming language
- **Scikit-learn**: Machine learning framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### MLOps & Infrastructure
- **DVC**: Data version control and pipeline orchestration
- **MLflow**: Experiment tracking and model registry
- **FastAPI**: High-performance API framework
- **Docker**: Containerization
- **GitHub Actions**: CI/CD pipeline

### Development & Quality
- **pytest**: Testing framework
- **Black**: Code formatting
- **Flake8**: Linting
- **Pydantic**: Data validation
- **Pre-commit**: Git hooks

## 📋 Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (optional)
- Git
- DVC (will be installed via requirements)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/basswala/diabetes-ml-pipeline.git
cd diabetes-ml-pipeline
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp env.template .env

# Edit .env with your credentials
nano .env
```

Required environment variables:
```env
MLFLOW_TRACKING_URI=https://dagshub.com/basswala/machinelearningpipeline.mlflow
MLFLOW_TRACKING_USERNAME=basswala
MLFLOW_TRACKING_PASSWORD=your_password
```

### 4. Run the Pipeline

```bash
# Initialize DVC (if not already done)
dvc init

# Pull data (if using remote storage)
dvc pull

# Run the complete pipeline
dvc repro
```

### 5. Start the API Server

```bash
# Start the FastAPI server
python -m src.api

# Or using uvicorn directly
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## 🐳 Docker Usage

### Build and Run

```bash
# Build the Docker image
docker build -t diabetes-ml-pipeline .

# Run with Docker Compose
docker-compose up -d

# Run standalone container
docker run -p 8000:8000 --env-file .env diabetes-ml-pipeline
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Lint and format
flake8 src/ tests/
black src/ tests/
```

## 📊 API Documentation

Once the API is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Example API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make a prediction
data = {
    "data": [{
        "pregnancies": 6,
        "glucose": 148,
        "blood_pressure": 72,
        "skin_thickness": 35,
        "insulin": 0,
        "bmi": 33.6,
        "diabetes_pedigree_function": 0.627,
        "age": 50
    }]
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## 📁 Project Structure

```
diabetes-ml-pipeline/
├── src/                    # Source code
│   ├── __init__.py
│   ├── api.py             # FastAPI application
│   ├── config.py          # Configuration management
│   ├── schemas.py         # Data validation schemas
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Model training
│   └── evaluate.py        # Model evaluation
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── test_api.py        # API tests
│   └── test_schemas.py    # Schema tests
├── data/                  # Data directory
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── models/                # Trained models
├── .github/workflows/     # CI/CD pipelines
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile            # Docker image definition
├── dvc.yaml              # DVC pipeline definition
├── params.yaml           # Pipeline parameters
├── requirements.txt      # Python dependencies
├── pytest.ini           # pytest configuration
├── .pre-commit-config.yaml # Pre-commit hooks
└── README.md             # This file
```

## 🔄 CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline that:

1. **Runs tests** on multiple Python versions (3.9, 3.10, 3.11)
2. **Performs linting** with flake8 and black
3. **Generates coverage reports** and uploads to Codecov
4. **Builds Docker images** on main branch pushes
5. **Deploys to production** (configurable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install pre-commit hooks (`pre-commit install`)
4. Make your changes
5. Run tests (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Install pre-commit hooks
pre-commit install

# Install development dependencies
pip install -r requirements.txt

# Run tests before committing
pytest
```

## 📈 Monitoring and Observability

- **Health Checks**: Built-in health check endpoint
- **Logging**: Structured logging with configurable levels
- **Metrics**: MLflow tracking for model performance
- **Validation**: Automatic data validation with Pydantic

## 🔧 Configuration

All configuration is managed through environment variables. See `env.template` for all available options.

Key configuration areas:
- **MLflow**: Experiment tracking and model registry
- **API**: Server configuration
- **Model**: Model paths and versions
- **Data**: Input/output paths

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Pima Indians Diabetes Dataset
- MLflow community for excellent experiment tracking tools
- FastAPI team for the amazing web framework
- DVC team for data version control

---

**Made with ❤️ for the MLOps community**
