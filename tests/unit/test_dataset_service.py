from pathlib import Path

import pytest

from churn_service.core.exceptions import DatasetValidationError
from churn_service.services.dataset import DatasetService

DATASET_PATH = Path("data/churn_dataset.csv")


@pytest.fixture(scope="module")
def service() -> DatasetService:
    return DatasetService(DATASET_PATH)


def test_loads_real_csv_without_error(service: DatasetService) -> None:
    assert service is not None


def test_row_count_is_positive(service: DatasetService) -> None:
    assert service.get_info().row_count > 0


def test_preview_returns_correct_count(service: DatasetService) -> None:
    assert len(service.get_preview(5)) == 5


def test_preview_n_larger_than_dataset_returns_all_rows(service: DatasetService) -> None:
    info = service.get_info()
    assert len(service.get_preview(info.row_count + 1000)) == info.row_count


def test_feature_names_excludes_churn(service: DatasetService) -> None:
    assert "churn" not in service.get_info().feature_names


def test_column_count_is_feature_names_plus_churn(service: DatasetService) -> None:
    info = service.get_info()
    assert info.column_count == len(info.feature_names) + 1


def test_churn_distribution_sums_to_row_count(service: DatasetService) -> None:
    info = service.get_info()
    dist = info.churn_distribution
    assert dist.retained + dist.churned == info.row_count


def test_missing_columns_raises_dataset_validation_error(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("monthly_fee,usage_hours\n9.99,100\n")
    with pytest.raises(DatasetValidationError):
        DatasetService(bad_csv)


def test_missing_file_raises_file_not_found_error() -> None:
    with pytest.raises(FileNotFoundError):
        DatasetService(Path("nonexistent/path.csv"))


def test_get_dataframe_returns_dataframe_with_expected_shape(service: DatasetService) -> None:
    import pandas as pd

    df = service.get_dataframe()
    info = service.get_info()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (info.row_count, info.column_count)
