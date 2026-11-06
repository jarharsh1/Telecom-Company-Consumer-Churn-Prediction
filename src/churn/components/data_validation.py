"""Stage 2 - Data validation.

Checks the ingested (joined) frame against schema.yaml: the column set matches,
declared numeric columns are numeric-coercible (TotalCharges arrives as blank-
containing strings, which is tolerated as long as the non-blank values parse),
the target is present, complete, and binary. Writes a status.json to gate on.
"""

import pandas as pd

from churn.entity.config_entity import DataValidationConfig
from churn.logger import logger
from churn.utils.common import save_json


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate(self) -> bool:
        df = pd.read_csv(self.config.data_file)
        schema = self.config.schema
        target = schema["target_column"]
        # Expected columns = schema columns minus the already-dropped id column.
        expected = set(schema["columns"]) - {schema.get("id_column")}
        numeric_cols = schema["numeric_columns"]

        errors: list[str] = []

        # 1. Column set matches (id already dropped in ingestion).
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        if extra:
            errors.append(f"Unexpected columns: {sorted(extra)}")

        # 2. Declared numeric columns must be numeric-coercible. A blank string
        #    (TotalCharges has a few) is an allowed missing value, not an error;
        #    a non-blank, non-numeric value is.
        for col in numeric_cols:
            if col not in df.columns:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_blank = df[col].astype(str).str.strip() != ""
            unparsable = coerced.isna() & non_blank
            if unparsable.any():
                errors.append(
                    f"Numeric column '{col}' has {int(unparsable.sum())} non-numeric value(s)"
                )

        # 3. No completely empty columns.
        for col in df.columns:
            if df[col].isna().all():
                errors.append(f"Column '{col}' is entirely null")

        # 4. Target present, complete, binary.
        if target in df.columns:
            n_missing = int(df[target].isna().sum())
            if n_missing:
                errors.append(f"Target '{target}' has {n_missing} missing value(s)")
            labels = set(df[target].dropna().unique())
            if not labels.issubset({"Yes", "No"}):
                errors.append(f"Target '{target}' has unexpected labels: {labels}")

        status = {
            "validation_passed": len(errors) == 0,
            "n_rows": len(df),
            "n_columns": int(df.shape[1]),
            "errors": errors,
        }
        save_json(self.config.status_file, status)

        if errors:
            logger.error("Data validation FAILED: %s", errors)
        else:
            logger.info("Data validation passed (%d rows).", len(df))
        return status["validation_passed"]
