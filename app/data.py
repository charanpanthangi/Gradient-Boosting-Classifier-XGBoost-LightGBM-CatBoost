"""Data loading utilities for the breast cancer dataset.

This module keeps data loading in one place so that both the CLI pipeline
and the Jupyter notebook can reuse the same function.
"""
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the breast cancer dataset as pandas objects.

    Returns:
        Tuple containing the feature dataframe and target series.
    """
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    return X, y


__all__ = ["load_data"]
