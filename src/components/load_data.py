import pandas as pd
import sys
from typing import Tuple
from src.exception import CustomException
from src.logger import logging

def load_data(train_path: str, test_path: str = None, unseen_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load datasets from CSV files.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the test CSV file (optional).
        unseen_path (str): Path to the unseen CSV file (optional).

    Returns:
        tuple: (train_df, second_df) as pandas DataFrames, where second_df is test_df or unseen_df.

    Raises:
        CustomException: If file loading fails.
    """
    try:
        train_df = pd.read_csv(train_path)
        second_df = pd.read_csv(test_path) if test_path else pd.read_csv(unseen_path) if unseen_path else None
        return train_df, second_df
    except Exception as e:
        logging.info(f"Failed to load data from {train_path}, {test_path or unseen_path}")
        raise CustomException(e, sys)