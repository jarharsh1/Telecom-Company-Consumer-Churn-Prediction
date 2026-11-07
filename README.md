# Telecom Customer Churn — End-to-End MLOps

Predict whether a telecom customer is likely to **churn**, built as a complete,
reproducible MLOps project around the IBM Telco churn dataset (7043 customers,
19 features after joining, target `Churn`).

This repository started as an exploratory notebook
(`Telecom_Company_Customer_Churn.ipynb`) and has been rebuilt into a modular,
config-driven, testable, servable pipeline. The raw data ships as two
column-aligned halves (`TelcomCustomer-Churn_1.csv` + `_2.csv`) that the
ingestion stage joins back together.

## Architecture

```
   data/raw/TelcomCustomer-Churn_1.csv + _2.csv   (two halves, joined on ingest)
                     │
   ┌─────────────────▼──────────────────────────────────────────┐
   │                   Training pipeline (DVC DAG)                │
   │  1. Data Ingestion      join halves, drop customerID         │
   │  2. Data Validation     schema.yaml contract → status.json   │
   │  3. Data Transformation coerce TotalCharges, impute/scale/OHE│
   │  4. Model Trainer       GradientBoosting in sklearn Pipeline │
   │  5. Model Evaluation    metrics.json + MLflow run            │
   └─────────────────┬──────────────────────────────────────────┘
                     │  model.joblib (preprocessor + model bundled)
                     ▼
            FastAPI serving  (/health, /predict, /train)  → Docker
```

Driven by three files: `config/config.yaml` (paths), `params.yaml`
(hyperparameters), `schema.yaml` (data contract).

## Project layout

```
src/churn/
  components/     ingestion · validation · transformation · trainer · evaluation
  pipeline/       one runnable stage per component + prediction pipeline
  config/         ConfigurationManager (YAML → typed configs)
  entity/         frozen dataclass configs
  utils/          yaml/json/joblib helpers
  logger.py       rotating-file + stdout logging
  exception.py    error wrapper with file/line context
app/              FastAPI serving layer (schemas + endpoints)
tests/            unit (validation) + end-to-end (pipeline + API) tests
dvc.yaml          reproducible pipeline DAG
Dockerfile        trains at build, serves with uvicorn
.github/workflows CI: lint → train → test → build image
```

## Quickstart

```bash
pip install -r requirements.txt     # install (editable, with dev extras)
python main.py                      # run the full training pipeline
uvicorn app.main:app --reload       # serve the model (see /docs)
```

### Reproduce with DVC

```bash
dvc repro          # re-runs only the stages whose deps/params changed
dvc metrics show   # prints the evaluation metrics
```

### Experiment tracking (MLflow)

Each evaluation logs params + metrics + model to a local `mlruns/` by default
(set `model_evaluation.mlflow_uri` to point at a remote server):

```bash
mlflow ui          # browse runs at http://127.0.0.1:5000
```

### Docker

```bash
docker build -t telco-churn .
docker run -p 8000:8000 telco-churn
```

### Tests

```bash
pytest -q
```

## Example prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 79.85, "TotalCharges": "160.2"
  }'
```

```json
{ "churn_probability": 0.68, "prediction": "churn", "will_churn": true }
```

## Notes

- Churn is imbalanced (~27% positive); the evaluation threshold
  (`model_evaluation.decision_threshold`, default 0.4) favours recall, and the
  serving layer uses the same threshold so served labels match the metrics.
- `TotalCharges` arrives with a few blank strings; ingestion/validation tolerate
  them and transformation coerces + imputes them.
- Generated artifacts (`artifacts/`, `logs/`, `mlruns/`) are gitignored.
