from app.data import load_data
from app.evaluate import evaluate_models
from app.model import get_models, train_models
from app.preprocess import train_test_split_data


def test_evaluate_models_outputs_metrics():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=1)
    models = get_models(random_state=1)
    trained = train_models(models, X_train, y_train)
    metrics = evaluate_models(trained, X_test, y_test)

    for model_metrics in metrics.values():
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert 0 <= model_metrics[key] <= 1
        assert model_metrics["confusion_matrix"].shape == (2, 2)
