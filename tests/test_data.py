from app.data import load_data


def test_load_data_shapes():
    X, y = load_data()
    assert not X.empty
    assert len(X) == len(y)
    assert y.nunique() == 2
