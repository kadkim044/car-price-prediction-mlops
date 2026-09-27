from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def test_home():
    with patch("mlflow.sklearn.load_model") as mock_load_model:
        mock_load_model.return_value = MagicMock()

        from api import app

        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200
        assert response.json() == {
            "message": "Car Price Prediction API"
        }