import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.exception import CustomException
from src.logger import logging
import sys

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray, model_name: str) -> dict:
    """
    Compute various evaluation metrics for the model.

    Args:
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
        model_name (str): Name of the model.

    Returns:
        dict: Dictionary containing various metrics (accuracy, precision, recall, f1_macro, f1_weighted, roc_auc).

    Raises:
        CustomException: If metric computation fails.
    """
    try:
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Compute ROC-AUC if probabilities are provided
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:  # Binary classification check
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['roc_auc'] = None

        print(f"{model_name} Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.3f}")
        print(f"  Recall (Macro): {metrics['recall_macro']:.3f}")
        print(f"  F1-Score (Macro): {metrics['f1_macro']:.3f}")
        print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.3f}")
        if metrics['roc_auc'] is not None:
            print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
        else:
            print("  ROC-AUC: Not applicable (multi-class or no probabilities)")

        return metrics
    except Exception as e:
        logging.info(f"Error computing metrics for {model_name}")
        raise CustomException(e, sys)