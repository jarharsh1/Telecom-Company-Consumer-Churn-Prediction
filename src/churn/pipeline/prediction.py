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


@lru_cache(maxsize=1)
def _feature_columns() -> tuple:
    cfg = ConfigurationManager().get_data_transformation_config()
    return tuple(cfg.numeric_columns) + tuple(cfg.categorical_columns)


class PredictionPipeline:
    # The target is encoded Yes -> 1 (churn) in transformation, so the positive
    # class label is 1; we resolve its column in predict_proba from
    # ``model.classes_`` rather than hard-coding index 1.
    POSITIVE_LABEL = 1

    def __init__(self):
        self.model = _load_model()
        self.numeric_columns = _numeric_columns()
        self.threshold = _default_threshold()
        self.feature_columns = _feature_columns()
        self._pos_idx = list(self.model.classes_).index(self.POSITIVE_LABEL)

    def predict(self, features: dict, threshold: float | None = None) -> dict:
        if threshold is None:
            threshold = self.threshold
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required feature field(s): {missing}")
        # Reindex to the training feature order (extra keys ignored).
        df = pd.DataFrame([features]).reindex(columns=list(self.feature_columns))
        # Coerce numeric fields so a string like "" or "1234.5" is handled.
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Look up the positive-class column instead of assuming index 1, so a
        # reordered/retargeted model still reads the right probability.
        proba = float(self.model.predict_proba(df)[0, self._pos_idx])
        is_churn = proba >= threshold
        return {
            "churn_probability": round(proba, 4),
            "prediction": "churn" if is_churn else "retain",
            "will_churn": bool(is_churn),
        }
