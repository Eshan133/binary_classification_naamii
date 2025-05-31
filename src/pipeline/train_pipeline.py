import pandas as pd
import warnings
import sys
import os
import pickle  # Added for saving selected features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
from src.components.load_data import load_data
from src.components.preprocess_data import preprocess_data
from src.components.prepare_features_and_target import prepare_features_and_target
from src.components.drop_near_constant_features import drop_near_constant_features
from src.components.scale_features import scale_features
from src.components.select_features import select_features
from src.components.train_and_evaluate_model import train_and_evaluate_model
from src.components.perform_cross_validation import perform_cross_validation
from src.components.create_evaluation_report import create_evaluation_report
from src.exception import CustomException
from src.logger import logging

warnings.filterwarnings('ignore')

def main():
    try:
        print('----TRAINING HAS STARTED----')
        print('Loading datasets')
        # Load data
        train_df, test_df = load_data('artifacts/train_set.csv', test_path='artifacts/test_set.csv')

        print('Preprocessing')
        # Preprocess data
        train_df = preprocess_data(train_df, is_train=True)
        test_df = preprocess_data(test_df, is_train=False, train_df=train_df)

        # Prepare features and target
        X_train, y_train = prepare_features_and_target(train_df)
        X_test, y_test = prepare_features_and_target(test_df)

        # Drop near-constant features
        X_train, X_test = drop_near_constant_features(X_train, X_test)

        # Scale features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

        # Select features
        X_train_selected, X_test_selected, selected_features = select_features(X_train_scaled, y_train, X_test_scaled, n_features=150)

        # Save selected features
        features_filename = 'artifacts/selected_features.pkl'
        with open(features_filename, 'wb') as f:
            pickle.dump(selected_features, f)
        print(f"Selected features saved to '{features_filename}'")

        print('Preprocessing Completed')
        
        print('Define models and hyperparameter grids')
        # Define models and hyperparameter grids
        models = {
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'Logistic Regression': LogisticRegression(class_weight={0: 1, 1: 1.5}, random_state=42, max_iter=1000)
        }
        param_grids = {
            'Random Forest': {'n_estimators': [200, 300], 'max_depth': [None, 20], 'min_samples_split': [2, 5]},
            'XGBoost': {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [5, 7, 10], 'n_estimators': [200, 300]},
            'Logistic Regression': {'C': [0.5, 1, 2, 5, 10], 'solver': ['liblinear', 'lbfgs']}
        }

        print('Train and evaluate models')
        # Train and evaluate models
        best_models = {}
        metrics_dict = {}
        for name in models:
            best_model, metrics = train_and_evaluate_model(
                models[name], param_grids[name], X_train_selected, y_train, X_test_selected, y_test, name
            )
            best_models[name] = best_model
            metrics_dict[name] = metrics

        # Select the best model based on F1-macro
        best_model_name = max(metrics_dict, key=lambda name: metrics_dict[name]['f1_macro'])

        # Create evaluation report
        report_filename = 'artifacts/evaluation_report.txt'
        create_evaluation_report(metrics_dict, best_model_name, report_filename)

        # Save the best model
        best_model = best_models[best_model_name]
        print(f"\nBest Model: {best_model_name} with Macro Avg F1-score: {metrics_dict[best_model_name]['f1_macro']:.3f}")
        model_filename = f'artifacts/{best_model_name.lower().replace(" ", "_")}_model.pkl'
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        joblib.dump(best_model, model_filename)
        print(f"Best model saved as '{model_filename}'")
        print(f"Evaluation report saved as '{report_filename}'")

        # Perform cross-validation
        perform_cross_validation(best_models, X_train_selected, y_train)

    except CustomException as ce:
        logging.error(f"CustomException occurred: {ce}")
        raise
    except Exception as e:
        logging.error("Unexpected error occurred")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()