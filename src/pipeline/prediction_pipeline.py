import pandas as pd
import warnings
import sys
import os
import joblib
import pickle  # Added for loading selected features
from src.components.load_data import load_data
from src.components.preprocess_data import preprocess_data
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
        print('Loading datasets')
        # Load data (train_df for bounds, unseen_df for prediction)
        train_df, unseen_df = load_data('artifacts/train_set.csv', unseen_path='artifacts/blinded_test_set.csv')
        

        print('Preprocessing')
        # Preprocess data
        train_df = preprocess_data(train_df, is_train=True)
        unseen_df = preprocess_data(unseen_df, is_train=False, train_df=train_df)

        # Prepare features
        X_train = prepare_features(train_df)
        X_unseen = prepare_features(unseen_df)

        # Drop near-constant features using training data as reference
        X_train, X_unseen = drop_near_constant_features(X_train, X_unseen)

        # Scale features
        X_train_scaled, X_unseen_scaled = scale_features(X_train, X_unseen)

        # Load the saved selected features
        features_filename = 'artifacts/selected_features.pkl'
        with open(features_filename, 'rb') as f:
            selected_features = pickle.load(f)
        print(f"Loaded selected features from '{features_filename}'")

        # Select features using the saved feature list
        X_train_selected, X_unseen_selected, _ = select_features(X_train_scaled, y_train=None, X_other_scaled=X_unseen_scaled, n_features=150, selected_features=selected_features)

        # Load the saved best model
        model_filename = os.path.join('artifacts', [f for f in os.listdir('artifacts') if f.endswith('_model.pkl')][0])
        best_model = joblib.load(model_filename)
        print(f"Loaded model from '{model_filename}'")

        # Predict on unseen data
        predictions_df = predict_unseen(best_model, X_unseen_selected, unseen_df)

        # Save predictions
        output_filename = 'artifacts/unseen_predictions.csv'
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        predictions_df.to_csv(output_filename, index=False)
        print(f"Predictions saved to '{output_filename}'")
        print("Predictions for Unseen Dataset:")
        print(predictions_df)

    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
        raise
    except Exception as e:
        logging.error("Unexpected error occurred")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()