import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys

def preprocess_data(df: pd.DataFrame, is_train: bool = True, train_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Preprocess the dataset by dropping specified columns and clipping extreme values.

    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
        is_train (bool): Flag to indicate if this is the training set (to compute bounds).
        train_df (pd.DataFrame): Training DataFrame for bounds if not training set.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.

    Raises:
        CustomException: If preprocessing fails.
    """
    try:
        # Drop features 1712 to 1734
        features_to_drop = [f'Feature_{i}' for i in range(1712, 1735)]
        df = df.drop(columns=[col for col in features_to_drop if col in df.columns], errors='ignore')

        # Drop features to reduce multicollinearity
        cols_to_drop = ['Feature_2', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9', 'Feature_2032']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        # Clip extreme values (5th to 95th percentiles)
        numeric_cols = df.drop(columns=['ID', 'CLASS'], errors='ignore').columns
        for col in numeric_cols:
            if is_train:
                lower_bound, upper_bound = df[col].quantile([0.05, 0.95])
                df[f'{col}_lower_bound'] = lower_bound
                df[f'{col}_upper_bound'] = upper_bound
            else:
                lower_bound = train_df[f'{col}_lower_bound'].iloc[0]
                upper_bound = train_df[f'{col}_upper_bound'].iloc[0]
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    except Exception as e:
        logging.info("Error during data preprocessing")
        raise CustomException(e, sys)