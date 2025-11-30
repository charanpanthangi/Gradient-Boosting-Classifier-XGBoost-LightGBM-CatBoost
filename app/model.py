"""Model definitions for XGBoost, LightGBM, and CatBoost classifiers."""
from typing import Dict

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from xgboost import XGBClassifier


def get_models(random_state: int = 42) -> Dict[str, ClassifierMixin]:
    """Create a dictionary of the configured boosting models.

    Args:
        random_state: Seed for reproducibility where supported.

    Returns:
        Dictionary mapping model names to classifier instances.
    """
    xgb_clf = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=300,
        random_state=random_state,
    )

    lgbm_clf = LGBMClassifier(n_estimators=300, random_state=random_state)

    cat_clf = CatBoostClassifier(n_estimators=300, verbose=0, random_state=random_state)

    return {
        "xgboost": xgb_clf,
        "lightgbm": lgbm_clf,
        "catboost": cat_clf,
    }


def train_models(
    models: Dict[str, ClassifierMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, ClassifierMixin]:
    """Fit each model on the training data.

    Args:
        models: Dictionary of model instances.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Dictionary of trained models.
    """
    trained_models: Dict[str, ClassifierMixin] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


__all__ = ["get_models", "train_models"]
