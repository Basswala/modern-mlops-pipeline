import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
import logging
from typing import Tuple, Dict, Any
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

import mlflow

from config import config
from schemas import validate_dataframe

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set MLflow environment variables from config
if config.MLFLOW_TRACKING_USERNAME and config.MLFLOW_TRACKING_PASSWORD:
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config.MLFLOW_TRACKING_PASSWORD

def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict[str, Any]) -> GridSearchCV:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Parameter grid for tuning
        
    Returns:
        Fitted GridSearchCV object
    """
    logger.info("Starting hyperparameter tuning...")
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=2,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    return grid_search

## Load the parameters from params.yaml

params=yaml.safe_load(open("params.yaml"))["train"]

def train(data_path: str, model_path: str, random_state: int, n_estimators: int, max_depth: int) -> None:
    """
    Train a Random Forest classifier with hyperparameter tuning.
    
    Args:
        data_path: Path to the training data
        model_path: Path to save the trained model
        random_state: Random state for reproducibility
        n_estimators: Number of estimators for Random Forest
        max_depth: Maximum depth of the trees
    """
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        
        # Validate data
        data = validate_dataframe(data)
        
        X = data.drop(columns=["Outcome"])
        y = data['Outcome']
        
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Features: {list(X.columns)}")

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)


    ## start the MLFLOW run
    with mlflow.start_run():
        # #split the dataset into training and test sets
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
        signature=infer_signature(X_train,y_train)

        ## Define hyperparameter grid

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Perform hyperparameter tuning
        grid_search=hyperparameter_tuning(X_train,y_train,param_grid)

        ## get the best model
        best_model=grid_search.best_estimator_

        ## predict and evaluate the model

        y_pred=best_model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        print(f"Accuracy:{accuracy}")

        ## Log additional metrics \
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_param("best_n_estimatios",grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_sample_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        ## log the confusion matrix and classification report

        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)

        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model, "model",signature=signature)

        ## create the directory to save the model
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Load parameters from params.yaml
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)["train"]
        
        train(
            params['data'], 
            params['model'], 
            params['random_state'], 
            params['n_estimators'], 
            params['max_depth']
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        exit(1)








