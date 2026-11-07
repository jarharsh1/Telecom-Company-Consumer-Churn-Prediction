"""Stage 4 - Model training.

Loads the fitted preprocessor, fits a GradientBoosting classifier on the
training split, and saves a single serving artifact: an sklearn Pipeline
bundling preprocessor + model, so serving loads one object and accepts raw rows.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from churn.entity.config_entity import ModelTrainerConfig
from churn.logger import logger
from churn.utils.common import load_object, save_object


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self) -> str:
        train_df = pd.read_csv(self.config.train_file)
        target = self.config.target_column
        X = train_df.drop(columns=[target])
        y = train_df[target]

        preprocessor = load_object(self.config.preprocessor_file)

        model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
        )

        serving_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        serving_pipeline.fit(X, y)

        save_object(self.config.model_file, serving_pipeline)
        logger.info(
            "Trained GradientBoosting(n_estimators=%d, lr=%s, depth=%d) -> %s",
            self.config.n_estimators,
            self.config.learning_rate,
            self.config.max_depth,
            self.config.model_file,
        )
        return str(self.config.model_file)
