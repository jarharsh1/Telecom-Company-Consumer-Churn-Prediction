"""Stage 1 - Data ingestion.

The raw dataset ships as two column-aligned halves (demographics/services in
file 1, account info + target in file 2), one row per customer in the same
order. We join them column-wise, drop the identifier, and write a single
combined CSV for the downstream stages.
"""

import pandas as pd

from churn.entity.config_entity import DataIngestionConfig
from churn.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def ingest(self) -> str:
        f1, f2 = self.config.source_file_1, self.config.source_file_2
        for f in (f1, f2):
            if not f.exists():
                raise FileNotFoundError(f"Source data not found: {f}")

        left = pd.read_csv(f1)
        right = pd.read_csv(f2)
        if len(left) != len(right):
            raise ValueError(
                f"Row count mismatch between halves: {len(left)} vs {len(right)}"
            )

        combined = pd.concat(
            [left.reset_index(drop=True), right.reset_index(drop=True)], axis=1
        )

        # Drop the identifier if present; it carries no predictive signal.
        if self.config.id_column in combined.columns:
            combined = combined.drop(columns=[self.config.id_column])

        combined.to_csv(self.config.local_data_file, index=False)
        logger.info(
            "Ingested %d rows x %d cols -> %s",
            combined.shape[0],
            combined.shape[1],
            self.config.local_data_file,
        )
        return str(self.config.local_data_file)
