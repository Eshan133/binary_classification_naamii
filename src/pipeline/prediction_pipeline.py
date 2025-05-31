import pandas as pd
import warnings
import sys
import os
import joblib
from src.components.load_data import load_data
from src.components.preprocess_data import preprocess_data
from src.components.prepare_features_and_target import prepare_features_and_target
from src.components.prepare_features import prepare_features
from src.components.drop_near_constant_features import drop_near_constant_features
from src.components.scale_features import scale_features
from src.components.select_features import select_features
from src.components.predict_unseen import predict_unseen
from src.exception import CustomException
from src.logger import logging

warnings.filterwarnings('ignore')

def main():
    try:
        print('----PREDICTION HAS STARTED----')
        print('--Loading datasets')
        # Load data 
        train_df, unseen_df = load_data('artifacts/train_set.csv', unseen_path='artifacts/blinded_test_set.csv')

        print('--Preprocessing')
        # Preprocess data
        train_df = preprocess_data(train_df, is_train=True)
        unseen_df = preprocess_data(unseen_df, is_train=False, train_df=train_df)

        # Prepare features
        X_train, y_train = prepare_features_and_target(train_df)
        X_unseen = prepare_features(unseen_df)

        # Drop near-constant features using training data as reference
        X_train, X_unseen = drop_near_constant_features(X_train, X_unseen)

        # Scale features
        X_train_scaled, X_unseen_scaled = scale_features(X_train, X_unseen)

        # Select features (use the same number of features as training, 150)
        X_train_selected, X_unseen_selected = select_features(X_train_scaled, y_train, X_other_scaled=X_unseen_scaled, n_features=150)
        print('--Preprocessing Completed')

        # Load the saved best model
        model_filename = os.path.join('artifacts', [f for f in os.listdir('artifacts') if f.endswith('_model.pkl')][0])
        best_model = joblib.load(model_filename)
        print(f"Loaded model from '{model_filename}'")

        # Predict on unseen data
        print('--Prediction in action')
        predictions_df = predict_unseen(best_model, X_unseen_selected, unseen_df)

        print('Saving predictions')
        # Save predictions
        output_filename = 'artifacts/unseen_predictions.csv'
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        predictions_df.to_csv(output_filename, index=False)
        print(f"Predictions saved to '{output_filename}'")
        print("Predictions for Unseen Dataset:")
        print(predictions_df)

    except Exception as e:
        logging.error("Unexpected error occurred")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()