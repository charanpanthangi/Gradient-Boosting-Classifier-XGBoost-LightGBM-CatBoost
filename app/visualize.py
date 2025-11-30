"""Visualization helpers for model interpretation."""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import ClassifierMixin
from sklearn.metrics import ConfusionMatrixDisplay

sns.set_style("whitegrid")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_feature_importances(
    models: Dict[str, ClassifierMixin], feature_names: list[str], output_dir: str = "figures"
) -> dict[str, Path]:
    """Plot and save feature importances for each model.

    Args:
        models: Dictionary of trained models.
        feature_names: List of feature names to label the plots.
        output_dir: Directory to save the SVG files.

    Returns:
        Dictionary mapping model names to saved file paths.
    """
    output_paths: dict[str, Path] = {}
    output_path = Path(output_dir)
    _ensure_dir(output_path)

    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            # CatBoost has a dedicated method
            importances = model.get_feature_importance()

        sorted_idx = np.argsort(importances)[::-1]
        sorted_importances = np.array(importances)[sorted_idx]
        sorted_features = np.array(feature_names)[sorted_idx]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
        plt.title(f"Feature Importances - {name.capitalize()}")
        plt.tight_layout()
        save_path = output_path / f"feature_importances_{name}.svg"
        plt.savefig(save_path, format="svg")
        plt.close()
        output_paths[name] = save_path

    return output_paths


def plot_confusion_matrices(
    metrics: Dict[str, Dict[str, float | np.ndarray]], output_dir: str = "figures"
) -> dict[str, Path]:
    """Plot confusion matrices for each model.

    Args:
        metrics: Nested metrics dictionary from ``evaluate_models``.
        output_dir: Directory to save the SVG files.

    Returns:
        Dictionary mapping model names to saved file paths.
    """
    output_paths: dict[str, Path] = {}
    output_path = Path(output_dir)
    _ensure_dir(output_path)

    for name, model_metrics in metrics.items():
        cm = model_metrics["confusion_matrix"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title(f"Confusion Matrix - {name.capitalize()}")
        plt.tight_layout()
        save_path = output_path / f"confusion_matrix_{name}.svg"
        plt.savefig(save_path, format="svg")
        plt.close(fig)
        output_paths[name] = save_path

    return output_paths


__all__ = ["plot_feature_importances", "plot_confusion_matrices"]
