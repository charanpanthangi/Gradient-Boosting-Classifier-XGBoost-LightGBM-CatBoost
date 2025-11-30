"""Preprocessing helpers for splitting the dataset.

Boosted tree models do not require feature scaling to perform well because
splits are based on feature order, not magnitude. Scaling can still be useful
for distance-based models, but here we keep things simple and unscaled.
"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test partitions.

    Args:
        X: Feature dataframe.
        y: Target series.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed for reproducibility.

    Returns:
        Tuple containing training features, test features, training labels, test labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


__all__ = ["train_test_split_data"]
