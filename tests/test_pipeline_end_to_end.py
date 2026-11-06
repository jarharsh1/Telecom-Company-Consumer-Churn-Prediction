"""End-to-end smoke test: run the whole pipeline, then score via the API."""

import json

from fastapi.testclient import TestClient


def test_full_pipeline_and_api():
    from main import main as run_pipeline

    run_pipeline()

    with open("artifacts/model_evaluation/metrics.json") as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    # Telco churn is well-separated; a real model clears this easily.
    assert metrics["roc_auc"] > 0.75

    from app.main import app

    client = TestClient(app)
    assert client.get("/health").status_code == 200

    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": "160.2",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in {"churn", "retain"}
    assert 0.0 <= body["churn_probability"] <= 1.0
