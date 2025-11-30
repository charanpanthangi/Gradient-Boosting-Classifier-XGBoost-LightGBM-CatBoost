from app.data import load_data
from app.model import get_models, train_models
from app.preprocess import train_test_split_data


def test_models_train_and_predict():
    X, y = load_data()
    X_train, X_test, y_train, _ = train_test_split_data(X, y, test_size=0.2, random_state=0)

    models = get_models(random_state=0)
    trained = train_models(models, X_train, y_train)

    for name, model in trained.items():
        preds = model.predict(X_test)
        assert len(preds) == len(X_test), f"Model {name} did not return predictions for all samples"
