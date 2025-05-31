import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import GridSearchCV
from .compute_metrics import compute_metrics
from src.exception import CustomException
from src.logger import logging
import sys

def train_and_evaluate_model(model, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Tuple[object, dict]:
    """
    Train a model with GridSearchCV and evaluate on the test set.

    Args:
        model: Model instance to train.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model_name (str): Name of the model.

    Returns:
        Tuple[object, dict]: (best_model, metrics_dict) where best_model is the trained model and metrics_dict contains evaluation metrics.

    Raises:
        CustomException: If training or evaluation fails.
    """
    try:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        metrics = compute_metrics(y_test, y_pred, y_pred_proba, model_name)
        print(f"Best {model_name} Parameters: {grid.best_params_}")
        return best_model, metrics
    except Exception as e:
        logging.info(f"Error training/evaluating {model_name}")
        raise CustomException(e, sys)