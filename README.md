# Car Price Prediction — MLOps

An end-to-end machine learning project for predicting used-car prices, with model training, experiment tracking, API serving, monitoring, and automated retraining.

The project is designed as a practical MLOps pipeline rather than only a machine learning model.

## Architecture

```text
                    ┌──────────────────┐
                    │  Used Car Dataset│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Preparation │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Random Forest    │
                    │    Training      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      MLflow      │
                    │ Tracking + Model │
                    │     Registry     │
                    └────────┬─────────┘
                             │
                       Champion Model
                             │
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    │   /predict API   │
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
          ┌──────────────┐       ┌──────────────┐
          │ PostgreSQL  │       │   Prediction │
          │  Predictions│       │     Data     │
          └──────┬───────┘       └──────────────┘
                 │
                 ▼
          ┌──────────────┐
          │   Evidently  │
          │ Drift Monitor│
          └──────┬───────┘
                 │
          Drift ≥ 50%
                 │
                 ▼
          ┌──────────────┐
          │    Prefect   │
          │  Retraining  │
          └──────┬───────┘
                 │
                 └──────────► MLflow
```

## Tech Stack

- **Python**
- **Pandas** — data preparation
- **Scikit-learn** — machine learning pipeline
- **Random Forest** — price prediction model
- **MLflow** — experiment tracking and model registry
- **FastAPI** — prediction API
- **PostgreSQL** — prediction storage
- **Evidently** — data drift monitoring
- **Prefect** — workflow orchestration and retraining
- **Docker / Docker Compose** — containerization
- **Pytest** — automated testing
- **Ruff** — code quality and linting

## Project Structure

```text
car-price-prediction-mlops/
│
├── data/
│   └── raw/
│       └── used_car_dataset.csv
│
├── src/
│   ├── api.py
│   ├── data.py
│   ├── monitoring.py
│   ├── pipeline.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_train.py
│
├── Dockerfile
├── docker-compose.yml
├── prefect.yaml
├── pytest.ini
├── requirements.txt
└── README.md
```

## Machine Learning Pipeline

The dataset is loaded and prepared by `src/data.py`.

The preprocessing pipeline:

- removes duplicate rows
- converts mileage into numeric values
- converts prices into numeric values
- removes unrealistic mileage values
- removes the `Age` column
- separates features and target
- performs an 80/20 train-test split

The model uses:

- `Year`
- `kmDriven`
- `Brand`
- `model`
- `Transmission`
- `Owner`
- `FuelType`

Categorical features are encoded using `OneHotEncoder`, while missing numerical values are handled with median imputation.

The prediction model is a `RandomForestRegressor`.

## MLflow

Training runs are tracked with MLflow.

The experiment is:

```text
car-price-prediction
```

The trained model is registered as:

```text
car-price-model
```

The API loads the model using the `champion` alias:

```python
models:/car-price-model@champion
```

Start the MLflow server with:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --disable-security-middleware
```

MLflow is then available on port `5000`.

## Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd car-price-prediction-mlops
```

### 2. Create the virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install pytest httpx ruff
```

### 4. Start Docker services

```bash
docker compose up -d
```

Check the services:

```bash
docker compose ps
```

### 5. Start MLflow

In another terminal:

```bash
source .venv/bin/activate

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --disable-security-middleware
```

### 6. Start Prefect

In another terminal:

```bash
source .venv/bin/activate
prefect server start
```

Then configure the API:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

Start the worker:

```bash
prefect worker start --pool test
```

## Training the Model

Run:

```bash
PYTHONPATH=src python src/train.py
```

This trains the Random Forest model, calculates evaluation metrics, logs the run to MLflow, and registers the model as `car-price-model`.

## Prediction API

The FastAPI service runs on port `9696`.

Check that the API is running:

```bash
curl http://localhost:9696/
```

Expected response:

```json
{
  "message": "Car Price Prediction API"
}
```

### Make a prediction

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Brand": "Toyota",
    "model": "Innova",
    "Year": 2020,
    "kmDriven": 50000,
    "Transmission": "Manual",
    "Owner": "First Owner",
    "FuelType": "Diesel"
  }'
```

Example response:

```json
{
  "predicted_price": 123456.78
}
```

The prediction is also stored in PostgreSQL.

## Monitoring

`src/monitoring.py` compares the features of recent predictions with the reference test dataset using Evidently.

The monitored features are:

```text
Brand
model
Year
Transmission
kmDriven
Owner
FuelType
```

A monitoring report is generated as:

```text
monitoring_report.html
```

Run monitoring with:

```bash
PYTHONPATH=src python src/monitoring.py
```

The current drift threshold is **50% of columns**.

If at least 50% of the monitored columns are detected as drifted, the monitoring process triggers the Prefect training deployment.

## Automated Retraining

The Prefect deployment is:

```text
training-pipeline/car-price-training
```

When significant data drift is detected:

```text
Evidently
    ↓
Drift detected
    ↓
Prefect deployment
    ↓
Training pipeline
    ↓
New MLflow run
    ↓
New model version
```

The deployment can also be triggered manually:

```bash
prefect deployment run 'training-pipeline/car-price-training'
```

## Testing

The project contains tests for:

- data loading and preparation
- model pipeline creation
- FastAPI health endpoint

Run the test suite with:

```bash
python -m pytest
```

Expected result:

```text
3 passed
```

## Code Quality

Ruff is used for linting and import formatting.

Run:

```bash
ruff check src tests
```

The project currently passes Ruff checks.

## End-to-End Workflow

The complete workflow is:

```text
Raw used-car data
        ↓
Data preparation
        ↓
Train / test split
        ↓
Random Forest training
        ↓
MLflow experiment tracking
        ↓
Model Registry
        ↓
Champion model
        ↓
FastAPI prediction service
        ↓
PostgreSQL prediction storage
        ↓
Evidently monitoring
        ↓
Data drift detection
        ↓
Prefect retraining
        ↓
MLflow
```

## What This Project Demonstrates

This project demonstrates an end-to-end MLOps workflow including:

- reproducible data preparation
- machine learning training
- experiment tracking
- model registry
- model serving
- REST API development
- containerization
- database integration
- data drift monitoring
- workflow orchestration
- automated retraining
- automated testing
- code quality checks