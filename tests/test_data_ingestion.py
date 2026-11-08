"""Unit tests for the data-ingestion join guard.

These exercise ``DataIngestion._check_join`` directly with small synthetic
frames, so they need only pandas (no sklearn) and stay fast.
"""

from pathlib import Path

import pandas as pd
import pytest

from churn.components.data_ingestion import DataIngestion
from churn.entity.config_entity import DataIngestionConfig
from churn.utils.common import read_yaml

SCHEMA = read_yaml("schema.yaml")


def _ingestion() -> DataIngestion:
    config = DataIngestionConfig(
        root_dir=Path("."),
        source_file_1=Path("a.csv"),
        source_file_2=Path("b.csv"),
        id_column=SCHEMA["id_column"],
        local_data_file=Path("out.csv"),
        schema=SCHEMA,
    )
    return DataIngestion(config)


def _joined_frame() -> pd.DataFrame:
    """A well-formed joined frame: every schema column plus the id column."""
    cols = list(SCHEMA["columns"]) + [SCHEMA["id_column"]]
    return pd.DataFrame({c: [0, 1] for c in cols})


def test_check_join_accepts_well_formed_frame():
    _ingestion()._check_join(_joined_frame())  # should not raise


def test_check_join_rejects_duplicate_columns():
    df = _joined_frame()
    # Simulate overlapping halves: a duplicated column name.
    dup = pd.concat([df, df[["gender"]]], axis=1)
    with pytest.raises(ValueError, match="Duplicate columns"):
        _ingestion()._check_join(dup)


def test_check_join_rejects_missing_expected_column():
    df = _joined_frame().drop(columns=["Churn"])
    with pytest.raises(ValueError, match="missing expected columns"):
        _ingestion()._check_join(df)
