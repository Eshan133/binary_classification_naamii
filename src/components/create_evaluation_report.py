from src.exception import CustomException
from src.logger import logging
import sys

def create_evaluation_report(metrics_dict: dict, best_model_name: str, report_path: str) -> None:
    """
    Create an evaluation report with all metrics for each model and save it to the specified path.

    Args:
        metrics_dict (dict): Dictionary with model names as keys and their metrics as values.
        best_model_name (str): Name of the best model based on F1-macro score.
        report_path (str): File path where the report will be saved.

    Raises:
        CustomException: If report creation fails.
    """
    try:
        with open(report_path, 'w') as f:
            f.write("Model Evaluation Report\n")
            f.write("======================\n\n")
            f.write("Model Name          | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) | ROC-AUC\n")
            f.write("--------------------|----------|------------------|---------------|------------|---------------|---------\n")
            for name, metrics in metrics_dict.items():
                roc_auc = metrics['roc_auc'] if metrics['roc_auc'] is not None else "N/A"
                f.write(f"{name:<20} | {metrics['accuracy']:.3f}   | {metrics['precision_macro']:.3f}           | {metrics['recall_macro']:.3f}        | {metrics['f1_macro']:.3f}     | {metrics['f1_weighted']:.3f}      | {roc_auc}\n")

            # Add best model summary
            best_metrics = metrics_dict[best_model_name]
            f.write("\nBest Model (based on F1-Macro):\n")
            f.write(f"  Name: {best_model_name}\n")
            f.write(f"  Accuracy: {best_metrics['accuracy']:.3f}\n")
            f.write(f"  Precision (Macro): {best_metrics['precision_macro']:.3f}\n")
            f.write(f"  Recall (Macro): {best_metrics['recall_macro']:.3f}\n")
            f.write(f"  F1-Score (Macro): {best_metrics['f1_macro']:.3f}\n")
            f.write(f"  F1-Score (Weighted): {best_metrics['f1_weighted']:.3f}\n")
            roc_auc = best_metrics['roc_auc'] if best_metrics['roc_auc'] is not None else "N/A"
            f.write(f"  ROC-AUC: {roc_auc}\n")

        logging.info(f"Evaluation report saved to {report_path}")
    except Exception as e:
        logging.error("Error creating evaluation report")
        raise CustomException(e, sys)