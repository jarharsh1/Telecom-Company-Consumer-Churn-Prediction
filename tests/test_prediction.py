"""Unit tests for the prediction pipeline's input validation.

These bypass ``__init__`` (which would load a trained model from disk) and set
only the attributes the guard needs, so they run without the full pipeline.
"""

import pytest

from churn.pipeline.prediction import PredictionPipeline


def _pipeline_with_columns(columns):
    p = PredictionPipeline.__new__(PredictionPipeline)
    p.feature_columns = tuple(columns)
    p.numeric_columns = ()
    p.threshold = 0.4
    p.model = None  # must not be reached when a field is missing
    return p


def test_missing_feature_raises_clear_error():
    p = _pipeline_with_columns(["tenure", "MonthlyCharges"])
    with pytest.raises(ValueError, match="Missing required feature field"):
        p.predict({"tenure": 12})  # 'MonthlyCharges' missing


def test_missing_feature_names_the_field():
    p = _pipeline_with_columns(["tenure", "MonthlyCharges"])
    with pytest.raises(ValueError, match="MonthlyCharges"):
        p.predict({"tenure": 12})
