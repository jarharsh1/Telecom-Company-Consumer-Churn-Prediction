"""Stage 1 - Data ingestion.

The raw dataset ships as two column-aligned halves (demographics/services in
file 1, account info + target in file 2), one row per customer in the same
order. We join them column-wise, drop the identifier, and write a single
combined CSV for the downstream stages.
"""

from collections import Counter

import pandas as pd

from churn.entity.config_entity import DataIngestionConfig
from churn.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _check_join(self, combined: pd.DataFrame) -> None:
        """Guard the positional column-join.

        Because the two halves are concatenated by row position with no shared
        key, a misaligned or overlapping export would otherwise pass silently.
        We reject duplicate column names (a sign the halves overlapped) and, when
        a schema is available, require every expected joined column to be present
        (so a dropped target/key column fails loudly here, not three stages later).
        """
        dupes = sorted(c for c, n in Counter(combined.columns).items() if n > 1)
        if dupes:
            raise ValueError(
                f"Duplicate columns after joining the two halves: {dupes}. "
                "The halves appear to overlap or be misaligned."
            )

        schema = self.config.schema or {}
        schema_cols = schema.get("columns")
        if schema_cols:
            # schema['columns'] describes the id-dropped frame; the raw join may
            # still carry the id column, so allow it in the expected set.
            expected = set(schema_cols) | {self.config.id_column}
            actual = set(combined.columns)
            missing = expected - actual
            if missing:
                raise ValueError(
                    f"Joined frame is missing expected columns: {sorted(missing)}. "
                    "The source halves do not match the schema contract."
                )

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

        # The halves are joined purely by row position (there is no shared key:
        # file 2 has no customerID), so harden the "same order / disjoint
        # columns" contract before anything downstream trusts the join.
        self._check_join(combined)

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
