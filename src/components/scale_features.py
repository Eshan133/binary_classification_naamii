import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
import sys

def scale_features(X_train: pd.DataFrame, X_other: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_other (pd.DataFrame): Test or unseen feature matrix (optional).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (X_train_scaled, X_other_scaled) as scaled DataFrames.

    Raises:
        CustomException: If feature scaling fails.
    """
    try:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        
        if X_other is not None:
            # Align X_other with X_train's columns
            X_other_aligned, _ = X_other.align(X_train, join='right', axis=1, fill_value=0)
            X_other_scaled = pd.DataFrame(scaler.transform(X_other_aligned), columns=X_train.columns)
            
            for X, name in [(X_train_scaled, 'train'), (X_other_scaled, 'test' if 'CLASS' in X_other.columns else 'unseen')]:
                if np.any(np.isinf(X)) or np.any(np.isnan(X)):
                    print(f"Infinite or NaN values found in {name} set after scaling. Replacing with 0...")
                    X.replace([np.inf, -np.inf], 0, inplace=True)
                    X.fillna(0, inplace=True)
            return X_train_scaled, X_other_scaled
        return X_train_scaled, None
    except Exception as e:
        logging.info("Error scaling features")
        raise CustomException(e, sys)