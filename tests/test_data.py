from data import load_and_prepare_data


def test_load_and_prepare_data():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    expected_features = [
        "Brand",
        "model",
        "Year",
        "Transmission",
        "kmDriven",
        "Owner",
        "FuelType",
    ]
    assert list(X_train.columns) == expected_features
    assert list(X_test.columns) == expected_features