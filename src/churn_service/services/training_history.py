from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from churn_service.core.exceptions import HistoryLoadError, HistoryWriteError
from churn_service.schemas.history import TrainingRecord

logger = logging.getLogger(__name__)

_HISTORY_FILENAME = "training_history.json"


class TrainingHistoryService:
    """Owns persistence of training run history in training_history.json.

    Records are stored newest-first: each append prepends the new record so
    load() returns them in the same order without reversal.
    """

    def __init__(self, models_dir: Path) -> None:
        self._path = models_dir / _HISTORY_FILENAME

    def append(self, record: TrainingRecord) -> None:
        records = self._read_raw()
        records.insert(0, record.model_dump(mode="json"))
        self._write_raw(records)
        logger.info(
            "History record appended: model_type=%s trained_at=%s",
            record.model_type,
            record.trained_at,
        )

    def load(self, model_type: str | None = None, limit: int = 10) -> list[TrainingRecord]:
        raw = self._read_raw()
        try:
            records: list[TrainingRecord] = [TrainingRecord.model_validate(r) for r in raw]
        except ValidationError as exc:
            raise HistoryLoadError(
                f"Training history contains invalid records: {self._path}"
            ) from exc
        if model_type is not None:
            records = [r for r in records if r.model_type == model_type]
        return records[:limit]

    def _read_raw(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            raise HistoryLoadError(
                f"Failed to parse training history: {self._path}"
            ) from exc
        if not isinstance(data, list):
            raise HistoryLoadError(
                f"Expected a JSON array in training history: {self._path}"
            )
        return data

    def _write_raw(self, records: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(records, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            tmp.rename(self._path)
        except OSError as exc:
            raise HistoryWriteError(
                f"Failed to write training history: {self._path}"
            ) from exc
