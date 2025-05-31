import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
from typing import Tuple  # Import Tuple for type hints


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target for training/testing.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) where X is the feature matrix and y is the target series.

    Raises:
        CustomException: If feature preparation fails.
    """
    try:
        X = df.drop(columns=['ID', 'CLASS', *[f'{col}_lower_bound' for col in df.columns if 'lower_bound' in col],
                            *[f'{col}_upper_bound' for col in df.columns if 'upper_bound' in col]], errors='ignore')
        y = df['CLASS'] if 'CLASS' in df.columns else None
        return X, y
    except Exception as e:
        logging.info("Error preparing features and target")
        raise CustomException(e, sys)