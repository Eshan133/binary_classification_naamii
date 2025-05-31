import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
from typing import Tuple

def drop_near_constant_features(X_train: pd.DataFrame, X_other: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop features with near-constant values based on training data.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_other (pd.DataFrame): Test or unseen feature matrix (optional).

    Returns:
        tuple: (X_train, X_other) with near-constant features removed.

    Raises:
        CustomException: If feature dropping fails.
    """
    try:
        stds = X_train.std()
        low_variance_cols = stds[stds < 1e-6].index
        print(f"Dropping {len(low_variance_cols)} near-constant features: {low_variance_cols}")
        X_train = X_train.drop(columns=low_variance_cols)
        if X_other is not None:
            X_other = X_other.drop(columns=[col for col in low_variance_cols if col in X_other.columns])
        return X_train, X_other
    except Exception as e:
        logging.info("Error dropping near-constant features")
        raise CustomException(e, sys)