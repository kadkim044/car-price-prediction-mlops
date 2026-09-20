from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn


app = FastAPI(title="Car Price Prediction API")


mlflow.set_tracking_uri("http://host.docker.internal:5000")
model = mlflow.sklearn.load_model(
    "models:/car-price-model@champion"
)
class Car(BaseModel):
    Brand: str
    model: str
    Year: int
    kmDriven: float | None = None
    Transmission: str
    Owner: str
    FuelType: str
    
@app.get("/")
def home():
    return {"message": "Car Price Prediction API"}
@app.post("/predict")
def predict(car:Car):
    data=pd.DataFrame([car.model_dump()])
    prediction = model.predict(data)[0]

    return {
        "predicted_price": float(prediction)
    }