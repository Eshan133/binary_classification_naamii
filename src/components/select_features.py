import pandas as pd
import pickle
from typing import Tuple, List
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomException
from src.logger import logging
import sys

def select_features(X_train_scaled: pd.DataFrame, y_train: pd.Series = None, X_other_scaled: pd.DataFrame = None, n_features: int = 150, selected_features: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select top features based on Random Forest importance or pre-selected features.

    Args:
        X_train_scaled (pd.DataFrame): Scaled training feature matrix.
        y_train (pd.Series, optional): Training labels. If None, selected_features must be provided.
        X_other_scaled (pd.DataFrame, optional): Scaled test or unseen feature matrix.
        n_features (int): Number of top features to select (if y_train is provided).
        selected_features (List[str], optional): Pre-selected feature names to use if y_train is None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (X_train_selected, X_other_selected) with selected features.

    Raises:
        CustomException: If feature selection fails.
    """
    try:
        if y_train is not None:
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X_train_scaled, y_train)
            feature_importance = pd.Series(rf.feature_importances_, index=X_train_scaled.columns)
            top_features = feature_importance.nlargest(n_features).index.tolist()
            print("Top 10 Feature Importances:")
            print(feature_importance.nlargest(10))
        else:
        
            if selected_features is None:
                raise ValueError("y_train is None, but no selected_features provided.")
            top_features = selected_features

        X_train_selected = X_train_scaled[top_features]
        X_other_selected = X_other_scaled[top_features] if X_other_scaled is not None else None
        return X_train_selected, X_other_selected, top_features
    except Exception as e:
        logging.info("Error during feature selection")
        raise CustomException(e, sys)