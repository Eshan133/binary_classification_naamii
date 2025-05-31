import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.logger import logging
import sys

def select_features(X_train_scaled: pd.DataFrame, y_train: pd.Series, X_other_scaled: pd.DataFrame = None, n_features: int = 150) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select top features based on Random Forest importance.

    Args:
        X_train_scaled (pd.DataFrame): Scaled training feature matrix.
        y_train (pd.Series): Training labels.
        X_other_scaled (pd.DataFrame): Scaled test or unseen feature matrix (optional).
        n_features (int): Number of top features to select.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (X_train_selected, X_other_selected) with selected features.

    Raises:
        CustomException: If feature selection fails.
    """
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train_scaled, y_train)
        feature_importance = pd.Series(rf.feature_importances_, index=X_train_scaled.columns)
        top_features = feature_importance.nlargest(n_features).index
        print("Top 10 Feature Importances:")
        print(feature_importance.nlargest(10))
        X_train_selected = X_train_scaled[top_features]
        X_other_selected = X_other_scaled[top_features] if X_other_scaled is not None else None
        return X_train_selected, X_other_selected
    except Exception as e:
        logging.info("Error during feature selection")
        raise CustomException(e, sys)