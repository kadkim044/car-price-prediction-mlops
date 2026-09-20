from prefect import flow,task
from data import load_and_prepare_data
from train import create_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("car-price-prediction")

@task
def load_data():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    return X_train, X_test, y_train, y_test
    
@task
def train_model(X_train,y_train):
    model=create_model()
    model.fit(X_train,y_train)
    return model

@task
def evaluate_model(model_pipeline,X_test,y_test):
    y_pred = model_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)


    print("Returned metrics:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R²:", r2)
    return mae, rmse, r2


@flow
def training_pipeline():
    X_train, X_test, y_train, y_test=load_data()
    with mlflow.start_run():
        model_pipeline=train_model(X_train,y_train)
        mae, rmse, r2 = evaluate_model(
            model_pipeline,
            X_test,
            y_test
        )
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
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
    print("\nMLflow run completed.")

if __name__ == "__main__":
    training_pipeline()