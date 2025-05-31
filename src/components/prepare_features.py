import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features by dropping ID and other columns for unseen data.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Feature matrix.

    Raises:
        CustomException: If feature preparation fails.
    """
    try:
        return df.drop(columns=['ID', *[f'{col}_lower_bound' for col in df.columns if 'lower_bound' in col],
                              *[f'{col}_upper_bound' for col in df.columns if 'upper_bound' in col]], errors='ignore')
    except Exception as e:
        logging.info("Error preparing features")
        raise CustomException(e, sys)