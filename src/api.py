import mlflow
import mlflow.sklearn
import pandas as pd
import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel

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
    conn = psycopg2.connect(
        host="host.docker.internal",
        port=5432,
        database="car_prices",
        user="postgres",
        password="postgres",
    )
    cursor=conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (
            brand,
            model,
            year,
            km_driven,
            transmission,
            owner,
            fuel_type,
            predicted_price
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            car.Brand,
            car.model,
            car.Year,
            car.kmDriven,
            car.Transmission,
            car.Owner,
            car.FuelType,
            float(prediction),
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return {
        "predicted_price": float(prediction)
    }