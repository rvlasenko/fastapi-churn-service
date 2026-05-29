from pathlib import Path

import pandas as pd

from churn_service.core.exceptions import DatasetValidationError
from churn_service.schemas.dataset import ChurnDistribution, DatasetInfoResponse
from churn_service.schemas.features import DatasetRowChurn

EXPECTED_COLUMNS: frozenset[str] = frozenset(DatasetRowChurn.model_fields.keys())


class DatasetService:
    def __init__(self, dataset_path: Path) -> None:
        self._df: pd.DataFrame = pd.read_csv(dataset_path)
        missing = EXPECTED_COLUMNS - set(self._df.columns)
        if missing:
            raise DatasetValidationError(f"Dataset is missing columns: {missing}")

    def get_preview(self, n: int) -> list[DatasetRowChurn]:
        rows = self._df.head(n).to_dict(orient="records")
        return [DatasetRowChurn.model_validate(row) for row in rows]

    def get_dataframe(self) -> pd.DataFrame:
        return self._df

    def get_info(self) -> DatasetInfoResponse:
        churn_counts = self._df["churn"].value_counts()
        feature_names = [col for col in self._df.columns if col != "churn"]
        return DatasetInfoResponse(
            row_count=len(self._df),
            column_count=len(self._df.columns),
            feature_names=feature_names,
            churn_distribution=ChurnDistribution(
                retained=int(churn_counts.get(0, 0)),
                churned=int(churn_counts.get(1, 0)),
            ),
        )
