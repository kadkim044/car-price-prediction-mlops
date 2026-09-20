from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from data import load_and_prepare_data

import mlflow


numeric_features = ["Year", "kmDriven"]

categorical_features = [
    "Brand",
    "model",
    "Transmission",
    "Owner",
    "FuelType",
]


def create_model():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    return model_pipeline


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("car-price-prediction")


if __name__ == "__main__":

    with mlflow.start_run():

        model_pipeline = create_model()

        mlflow.set_tag(
            "model_type",
            "RandomForestRegressor"
        )

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        X_train, X_test, y_train, y_test = load_and_prepare_data()

        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(
            model_pipeline,
            name="model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
            registered_model_name="car-price-model",
        )

        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R²:", r2)