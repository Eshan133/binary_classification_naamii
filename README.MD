# Task 2 - Data Classification

## Overview

This project, "Task 2 - Data Classification," focuses on building a machine learning pipeline to classify data using a dataset provided in CSV format. The pipeline includes EDA, data preprocessing, feature engineering, model training, evaluation, and prediction on unseen data.

The pipeline is split into two main scripts:

- `train_pipeline.py`: Handles data loading, preprocessing, feature selection, model training, evaluation, and saving the best model.
- `prediction_pipeline.py`: Loads the trained model and preprocessed unseen data to generate predictions.

## Project Structure

The project is organized as follows:

```
code/
├── artifacts/
│   ├── train_set.csv           # Training dataset
│   ├── test_set.csv            # Test dataset
│   ├── unseen_set.csv          # Unseen dataset for predictions
│   ├── selected_features.pkl   # Saved selected features from training
│   ├── evaluation_report.txt   # Evaluation report with metrics for all models
│   ├── *_model.pkl             # Saved best model (e.g., logistic_regression_model.pkl)
│   └── unseen_predictions.csv  # Predictions on unseen data
│
├── src/
│   ├── components/
│   │   ├── compute_metrics.py           # Computes evaluation metrics
│   │   ├── create_evaluation_report.py  # Generates evaluation report
│   │   ├── drop_near_constant_features.py  # Drops near-constant features
│   │   ├── load_data.py                # Loads datasets
│   │   ├── predict_unseen.py           # Predicts on unseen data
│   │   ├── prepare_features.py         # Prepares features for modeling
│   │   ├── prepare_features_and_target.py  # Prepares features and target for training
│   │   ├── preprocess_data.py          # Preprocesses data (e.g., clipping extreme values)
│   │   ├── scale_features.py           # Scales features
│   │   ├── select_features.py          # Selects top features
│   │   ├── train_and_evaluate_model.py # Trains and evaluates models
│   │   └── perform_cross_validation.py # Performs cross-validation
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Main training pipeline script
│   │   └── prediction_pipeline.py # Main prediction pipeline script
│   │
│   ├── exception.py  # Custom exception handling
│   │
│   └── logger.py     # Logging utility
│
├── task2/           # Virtual environment directory
│
├── requirements.txt  # Dependencies file with -e .
│
├── setup.py          # Setup configuration file
│
└── README.md        # This file
```

## Prerequisites

- **Python**: Version 3.8 or higher
- **Virtual Environment**: Recommended to isolate dependencies

### Setup Instructions

1. **Clone the Repository** (if applicable):

   ```bash
   git clone https://github.com/Eshan133/binary_classification_naamii.git
   cd Task 2-data classification
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv task2
   source task2/bin/activate  # On Windows: task2\Scripts\activate
   ```

3. **Install the Project and Dependencies**:

   - The `requirements.txt` file contains `-e .`, which installs the project in editable mode and runs `setup.py`.
   - Install the project and its dependencies:

     ```bash
     pip install -r requirements.txt
     ```
   - This will install the required packages (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`) and set up the `train-pipeline` and `predict-pipeline` command-line scripts.

4. **Prepare the Data**:

   - Ensure the `artifacts/` directory contains the following files:
     - `train_set.csv`: Training data with features and `CLASS` labels.
     - `test_set.csv`: Test data with features and `CLASS` labels.
     - `blinded_test_set.csv`: Unseen data with features (without `CLASS` labels).

## Usage

The project consists of two main pipelines: training and prediction, accessible as command-line scripts after installation.

### 1. Training Pipeline

The training pipeline (`train_pipeline.py`) performs the following steps:

- Loads `train_set.csv` and `test_set.csv` from `artifacts/`.
- Preprocesses the data (drops specified features, clips extreme values using 5th and 95th percentiles).
- Prepares features and targets, drops near-constant features, scales features, and selects the top 150 features using Random Forest importance.
- Trains three models: Random Forest, XGBoost, and Logistic Regression, using GridSearchCV for hyperparameter tuning.
- Evaluates models using multiple metrics (Accuracy, Precision, Recall, F1-Macro, F1-Weighted, ROC-AUC).
- Saves the best model (based on F1-Macro score) to `artifacts/` (e.g., `logistic_regression_model.pkl`).
- Saves the selected features to `artifacts/selected_features.pkl`.
- Generates an evaluation report in `artifacts/evaluation_report.txt`.

#### Run the Training Pipeline

From the project root directory, execute:

```bash
python src/pipeline/train_pipeline.py
```

#### Output

- **Best Model**: Saved as `artifacts/<best_model_name>_model.pkl`.
- **Selected Features**: Saved as `artifacts/selected_features.pkl`.
- **Evaluation Report**: Saved as `artifacts/evaluation_report.txt`, containing metrics for all models.

### 2. Prediction Pipeline

The prediction pipeline (`prediction_pipeline.py`) performs the following steps:

- Loads `train_set.csv` (for preprocessing consistency) and `blinded_test_set.csv` from `artifacts/`.
- Preprocesses the data using the same steps as the training pipeline.
- Prepares features, drops near-constant features, scales features, and selects features using the saved `selected_features.pkl`.
- Loads the best model from `artifacts/`.
- Generates predictions on the blinded test data.
- Saves predictions to `artifacts/unseen_predictions.csv`.

#### Run the Prediction Pipeline

Ensure the training pipeline has been run first to generate the model and selected features. Then, from the project root directory, execute:

```bash
python src/pipeline/prediction_pipeline.py
```

#### Output

- **Predictions**: Saved as `artifacts/unseen_predictions.csv`, containing `ID` and predicted `CLASS` columns.

## Notes

- **Error Handling**: The project includes custom exception handling (`exception.py`) and logging (`logger.py`) to debug issues.
- **Customization**: To modify the number of features selected, we can update the `n_features` parameter in both pipelines (currently set to 150).

## Troubleshooting

- **FileNotFoundError**: Ensure all CSV files are in the `artifacts/` directory and paths are correct.
- **Feature Mismatch**: If the prediction pipeline fails due to feature mismatches, verify that `selected_features.pkl` exists and matches the training setup.
- **Installation Issues**: If `pip install -r requirements.txt` fails, ensure `setup.py` is correctly configured and all dependencies are available.

## Future Improvements

- Add more evaluation metrics (e.g., confusion matrix, per-class metrics).
- Add support for additional models or feature selection methods.

## Acknowledgments

This project was developed with assistance from Grok, created by xAI, for debugging, code optimization, and documentation. Thanks Grok


#### Outputs are inside the artifacts folder
