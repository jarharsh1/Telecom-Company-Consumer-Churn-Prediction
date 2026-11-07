"""Stage 3 - Data transformation.

Coerces TotalCharges to numeric (blank strings -> NaN, imputed downstream),
encodes the target (Yes -> 1 = churn), builds a ColumnTransformer (impute+scale
numerics, impute+one-hot categoricals), and writes a stratified train/test
split plus the preprocessor fitted on the training split only.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.entity.config_entity import DataTransformationConfig
from churn.logger import logger
from churn.utils.common import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _coerce_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """TotalCharges (and any numeric col) may carry blank strings; coerce."""
        for col in self.config.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.config.numeric_columns),
                ("cat", categorical_pipe, self.config.categorical_columns),
            ]
        )

    def transform(self) -> tuple[str, str, str]:
        df = pd.read_csv(self.config.data_file)
        target = self.config.target_column

        df = self._coerce_numerics(df)

        # Never fabricate labels: drop rows with a missing target defensively.
        n_missing = int(df[target].isna().sum())
        if n_missing:
            logger.warning("Dropping %d row(s) with a missing target.", n_missing)
            df = df.dropna(subset=[target])

        # Encode target to 0/1 (Yes -> 1 = churn).
        df[target] = (df[target].astype(str).str.strip().str.lower() == "yes").astype(int)

        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df[target],
        )

        preprocessor = self._build_preprocessor()
        feature_cols = self.config.numeric_columns + self.config.categorical_columns
        preprocessor.fit(train_df[feature_cols])

        save_object(self.config.preprocessor_file, preprocessor)
        train_df.to_csv(self.config.train_file, index=False)
        test_df.to_csv(self.config.test_file, index=False)

        logger.info(
            "Transformation done: train=%d test=%d; preprocessor -> %s",
            len(train_df),
            len(test_df),
            self.config.preprocessor_file,
        )
        return (
            str(self.config.train_file),
            str(self.config.test_file),
            str(self.config.preprocessor_file),
        )
