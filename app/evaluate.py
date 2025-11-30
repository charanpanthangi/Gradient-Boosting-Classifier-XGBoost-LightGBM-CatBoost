"""Evaluation utilities for the boosting classifiers."""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_models(
    models: Dict[str, ClassifierMixin],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Dict[str, float | np.ndarray]]:
    """Compute performance metrics for each model.

    Args:
        models: Dictionary of trained models.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Nested dictionary of metrics for each model.
    """
    results: Dict[str, Dict[str, float | np.ndarray]] = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="binary", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="binary", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
    return results


__all__ = ["evaluate_models"]
