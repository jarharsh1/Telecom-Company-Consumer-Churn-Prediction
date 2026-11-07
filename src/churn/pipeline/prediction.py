"""Inference pipeline used by the serving layer.

Loads the trained serving pipeline once, coerces numeric fields (TotalCharges
may arrive as a string), and classifies at the configured decision threshold so
served labels match the evaluated metrics.
"""

from functools import lru_cache

import pandas as pd

from churn.config.configuration import ConfigurationManager
from churn.utils.common import load_object


@lru_cache(maxsize=1)
def _load_model():
    model_file = ConfigurationManager().get_model_trainer_config().model_file
    return load_object(model_file)


@lru_cache(maxsize=1)
def _numeric_columns() -> tuple:
    return tuple(ConfigurationManager().get_data_transformation_config().numeric_columns)


@lru_cache(maxsize=1)
def _default_threshold() -> float:
    return ConfigurationManager().get_model_evaluation_config().decision_threshold


class PredictionPipeline:
    def __init__(self):
        self.model = _load_model()
        self.numeric_columns = _numeric_columns()
        self.threshold = _default_threshold()

    def predict(self, features: dict, threshold: float | None = None) -> dict:
        if threshold is None:
            threshold = self.threshold
        df = pd.DataFrame([features])
        # Coerce numeric fields so a string like "" or "1234.5" is handled.
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        proba = float(self.model.predict_proba(df)[:, 1][0])
        is_churn = proba >= threshold
        return {
            "churn_probability": round(proba, 4),
            "prediction": "churn" if is_churn else "retain",
            "will_churn": bool(is_churn),
        }
