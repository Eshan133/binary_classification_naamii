import pandas as pd
from sklearn.model_selection import cross_val_score
from src.exception import CustomException
from src.logger import logging
import sys

def perform_cross_validation(models: dict, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Perform cross-validation for all models and print results.

    Args:
        models (dict): Dictionary of model names and their instances.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Raises:
        CustomException: If cross-validation fails.
    """
    try:
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
            print(f"\n{name} Cross-Validation F1-Scores (on training data): {cv_scores}")
            print(f"Average CV F1-Score (macro avg) for {name}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    except Exception as e:
        logging.info("Error during cross-validation")
        raise CustomException(e, sys)