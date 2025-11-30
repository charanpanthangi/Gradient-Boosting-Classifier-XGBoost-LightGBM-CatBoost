# Gradient Boosting Classifier: XGBoost, LightGBM, CatBoost

Beginner-friendly template and tutorial for training, comparing, and visualizing
three popular gradient boosting classifiers on the scikit-learn breast cancer
dataset.

## What is boosting?
Boosting is an ensemble technique that trains weak learners (often shallow
decision trees) sequentially. Each new tree focuses on correcting the errors of
the previous ensemble, leading to a strong model that outperforms a single tree
by reducing bias and variance.

## Why boosted trees work so well
- **Iterative error correction:** Each tree concentrates on misclassified
  samples from prior iterations.
- **Built-in feature selection:** Tree splits naturally prioritize informative
  features.
- **Non-linear decision boundaries:** Ensembles of trees capture complex
  patterns without heavy feature engineering.

## XGBoost vs. LightGBM vs. CatBoost
- **XGBoost:** Highly configurable with regularization and robust handling of
  sparse inputs; great general-purpose choice.
- **LightGBM:** Uses histogram-based splitting and leaf-wise growth, making it
  very fast on large, high-dimensional datasets.
- **CatBoost:** Handles categorical features elegantly (ordered boosting);
  strong defaults and less need for extensive tuning.

## When to choose which model
- Use **XGBoost** when you need a battle-tested default with extensive control.
- Choose **LightGBM** for speed on large datasets with many features.
- Try **CatBoost** when you have categorical data or want strong out-of-the-box
  performance with minimal tuning.

## Dataset
The project uses the scikit-learn `load_breast_cancer` dataset: 30 numeric
features predicting whether a tumor is malignant or benign.

## Project structure
```
<repo-root>/
├── app/
│   ├── data.py          # Load the dataset
│   ├── preprocess.py    # Train/test split
│   ├── model.py         # Model definitions and training
│   ├── evaluate.py      # Metrics for each model
│   ├── visualize.py     # Feature importances + confusion matrices
│   ├── main.py          # End-to-end pipeline
├── notebooks/
│   └── demo_gradient_boosting_classifier.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_evaluate.py
├── examples/
│   └── README_examples.md
├── requirements.txt
├── Dockerfile
├── .gitignore
└── LICENSE
```

## How the pipeline works
1. **Load data** from scikit-learn.
2. **Split** into training and test sets (scaling is optional for tree models).
3. **Train** three classifiers with 300 estimators each: XGBoost, LightGBM,
   CatBoost.
4. **Evaluate** using accuracy, precision, recall, F1-score, and confusion
   matrices.
5. **Visualize** feature importances and confusion matrices (SVG files saved to
   `figures/`).
6. **Compare** models in a tidy summary table.

## Quickstart
```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python app/main.py

# Launch the demo notebook
jupyter notebook notebooks/demo_gradient_boosting_classifier.ipynb
```

## Docker
```bash
docker build -t boosting-classifier .
docker run --rm boosting-classifier
```

## Tests
```bash
pytest
```

## Future improvements
- Early stopping to reduce overfitting and speed up training.
- Hyperparameter tuning (GridSearchCV/Optuna) for optimal performance.
- SHAP-based explainability for feature impact analysis.

## License
MIT License. See [LICENSE](LICENSE) for details.
