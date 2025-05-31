import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys

def predict_unseen(best_model, X_unseen_selected: pd.DataFrame, unseen_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict on unseen data and format the results.

    Args:
        best_model: Trained model to use for prediction.
        X_unseen_selected (pd.DataFrame): Preprocessed unseen features.
        unseen_df (pd.DataFrame): Original unseen DataFrame for ID column.

    Returns:
        pd.DataFrame: DataFrame with predictions.

    Raises:
        CustomException: If prediction fails.
    """
    try:
        y_pred_unseen = best_model.predict(X_unseen_selected)
        if 'ID' in unseen_df.columns:
            predictions_df = pd.DataFrame({
                'ID': unseen_df['ID'],
                'Predicted_CLASS': y_pred_unseen
            })
        else:
            predictions_df = pd.DataFrame({
                'Index': range(len(y_pred_unseen)),
                'Predicted_CLASS': y_pred_unseen
            })
        return predictions_df
    except Exception as e:
        logging.info("Error during prediction on unseen data")
        raise CustomException(e, sys)