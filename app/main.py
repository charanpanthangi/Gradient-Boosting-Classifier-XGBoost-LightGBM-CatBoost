"""Command-line entry point to run the full gradient boosting pipeline."""
from __future__ import annotations

import pandas as pd

from app.data import load_data
from app.evaluate import evaluate_models
from app.model import get_models, train_models
from app.preprocess import train_test_split_data
from app.visualize import plot_confusion_matrices, plot_feature_importances


def run_pipeline() -> pd.DataFrame:
    """Execute the end-to-end training, evaluation, and visualization pipeline.

    Returns:
        DataFrame summarizing model performance.
    """
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    print("Loaded dataset with shape:", X.shape)
    print("Training samples:", X_train.shape[0], "| Test samples:", X_test.shape[0])

    models = get_models()
    trained_models = train_models(models, X_train, y_train)

    metrics = evaluate_models(trained_models, X_test, y_test)

    feature_paths = plot_feature_importances(trained_models, list(X.columns))
    confusion_paths = plot_confusion_matrices(metrics)

    summary_rows = []
    for name, model_metrics in metrics.items():
        summary_rows.append(
            {
                "model": name,
                "accuracy": model_metrics["accuracy"],
                "precision": model_metrics["precision"],
                "recall": model_metrics["recall"],
                "f1": model_metrics["f1"],
                "feature_importances_svg": str(feature_paths[name]),
                "confusion_matrix_svg": str(confusion_paths[name]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\nModel comparison:")
    print(summary_df[['model', 'accuracy', 'precision', 'recall', 'f1']].sort_values(by="accuracy", ascending=False))

    return summary_df


if __name__ == "__main__":
    run_pipeline()
