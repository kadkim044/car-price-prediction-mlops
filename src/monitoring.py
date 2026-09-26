import psycopg2
import pandas as pd
from data import load_and_prepare_data
from evidently import Report
from evidently.presets import DataDriftPreset
from pipeline import training_pipeline

def load_reference_data():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    return X_test
def load_predictions():
    conn=psycopg2.connect(
        host="127.0.0.1",
        port=5432,
        database="car_prices",
        user="postgres",
        password="postgres"
    )
    cursor=conn.cursor()
    cursor.execute(
        """
            SELECT * FROM predictions;
        """
    )
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    conn.close()
    return df

def prepare_current_data(df):
    df=df.rename(
        columns={
            "brand": "Brand",
            "year": "Year",
            "km_driven": "kmDriven",
            "transmission": "Transmission",
            "owner": "Owner",
            "fuel_type": "FuelType",
        }
    )
    features = [
        "Brand",
        "model",
        "Year",
        "Transmission",
        "kmDriven",
        "Owner",
        "FuelType",
    ]

    return df[features]

def generate_drift_report(reference,current):
    report=Report(
        metrics=[
            DataDriftPreset(),
        ]
    )
    snapshot=report.run(
        reference_data=reference,
        current_data=current,
    )
    snapshot.save_html("monitoring_report.html")
    print("Monitoring report generated: monitoring_report.html")
    result = snapshot.dict()

    print("\nMonitoring results:")
    print(result)

    return result

def check_drift(result):
    drift_metric=next(
        metric
        for metric in result["metrics"]
        if metric["metric_name"].startswith("DriftedColumnsCount")
    )
    drift_share = drift_metric["value"]["share"]

    threshold = 0.5

    print(f"\nDrift share: {drift_share:.2%}")
    print(f"Threshold: {threshold:.2%}")

    if drift_share >= threshold:
        print("⚠️ Significant data drift detected!")
        return True

    print("✅ No significant data drift detected.")
    return False

def trigger_retraining():
    print("\n🚨 Triggering model retraining...")
    training_pipeline()
    print("✅ Retraining completed.")

if __name__ == "__main__":
    current = load_predictions()
    reference = load_reference_data()

    current = prepare_current_data(current)

    result = generate_drift_report(
        reference=reference,
        current=current,
    )

    drift_detected = check_drift(result)

    if drift_detected:
        trigger_retraining()