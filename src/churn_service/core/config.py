from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "churn-service"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    models_dir: Path = Path("models")
    dataset_path: Path = Path("data/churn_dataset.csv")
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                f"Invalid log_level: {v!r}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        return v_upper
