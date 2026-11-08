"""Unit tests for the schema-based data validation component."""

import pandas as pd

from churn.components.data_validation import DataValidation
from churn.entity.config_entity import DataValidationConfig
from churn.utils.common import load_json, read_yaml

SCHEMA = read_yaml("schema.yaml")


def _config(tmp_path, data_file):
    return DataValidationConfig(
        root_dir=tmp_path,
        data_file=data_file,
        status_file=tmp_path / "status.json",
        schema=SCHEMA,
    )


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": ["Female"],
            "SeniorCitizen": [0],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "tenure": [12],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["Fiber optic"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["Yes"],
            "StreamingMovies": ["Yes"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [79.85],
            "TotalCharges": ["955.1"],
            "Churn": ["No"],
        }
    )


def test_valid_data_passes(tmp_path):
    f = tmp_path / "ok.csv"
    _valid_frame().to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is True
    status = load_json(tmp_path / "status.json")
    assert status["validation_passed"] is True
    assert status["errors"] == []


def test_blank_totalcharges_is_tolerated(tmp_path):
    # A blank TotalCharges is an allowed missing value, not a validation error.
    df = _valid_frame()
    df["TotalCharges"] = [" "]
    f = tmp_path / "blank.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is True


def test_empty_totalcharges_field_is_tolerated(tmp_path):
    # One empty CSV field (reads back as NaN) alongside a valid value is an
    # allowed missing value, not a "non-numeric" error. Two rows so the column
    # is not entirely null.
    df = pd.concat([_valid_frame(), _valid_frame()], ignore_index=True)
    df.loc[0, "TotalCharges"] = None  # empty field -> NaN on read-back
    f = tmp_path / "empty.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is True


def test_non_numeric_totalcharges_fails(tmp_path):
    df = _valid_frame()
    df["TotalCharges"] = ["not-a-number"]
    f = tmp_path / "bad_num.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False


def test_missing_column_fails(tmp_path):
    df = _valid_frame().drop(columns=["tenure"])
    f = tmp_path / "missing.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False


def test_bad_target_labels_fail(tmp_path):
    df = _valid_frame()
    df["Churn"] = ["maybe"]
    f = tmp_path / "bad_target.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False


def test_categorical_column_arriving_numeric_fails(tmp_path):
    # A declared-object categorical column (Partner) that silently arrives as a
    # numeric dtype must be flagged by the dtype-family check.
    df = pd.concat([_valid_frame(), _valid_frame()], ignore_index=True)
    df["Partner"] = [0, 1]  # numeric instead of "Yes"/"No"
    f = tmp_path / "num_categorical.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False


def test_seniorcitizen_int_flag_is_allowed(tmp_path):
    # SeniorCitizen is declared int64 and intentionally numeric; the dtype-family
    # check must NOT flag it (regression guard for the categorical check above).
    df = pd.concat([_valid_frame(), _valid_frame()], ignore_index=True)
    df["SeniorCitizen"] = [0, 1]
    f = tmp_path / "senior.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is True
