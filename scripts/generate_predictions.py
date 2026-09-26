import pandas as pd
import requests

from data import load_and_prepare_data

API_URL = "http://127.0.0.1:9696/predict"

def generate_predictions(n=100):
    _,X_test,_,_=load_and_prepare_data()
    samples=X_test.sample(n=n,random_state=42)
    for i,(_,row) in enumerate(samples.iterrows(),start=1):
        payload = {
            "Brand": row["Brand"],
            "model": row["model"],
            "Year": int(row["Year"]),
            "kmDriven": (
                None
                if pd.isna(row["kmDriven"])
                else float(row["kmDriven"])
            ),
            "Transmission": row["Transmission"],
            "Owner": row["Owner"],
            "FuelType": row["FuelType"],
        }
        response = requests.post(API_URL, json=payload)

        if response.ok:
            prediction = response.json()["predicted_price"]
            print(f"[{i}/{n}] prediction: {prediction:.2f}")
        else:
            print(f"[{i}/{n}] ERROR {response.status_code}: {response.text}")
    
if __name__ == "__main__":
    generate_predictions(100)