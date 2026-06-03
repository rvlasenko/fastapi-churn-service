from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class TrainResponse(BaseModel):
    accuracy: float
    f1: float
    train_size: int
    test_size: int


class ModelType(StrEnum):
    LOGREG = "logreg"
    RANDOM_FOREST = "random_forest"


# Allowed hyperparameter names per model type.
ALLOWED_HYPERPARAMS: dict[ModelType, frozenset[str]] = {
    ModelType.LOGREG: frozenset(
        {
            "C",
            "max_iter",
            "class_weight",
            "solver",
            "random_state",
        }
    ),
    ModelType.RANDOM_FOREST: frozenset(
        {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "class_weight",
            "random_state",
        }
    ),
}

# ---------------------------------------------------------------------------
# Per-parameter value validators
# Each returns None on success or an error message string on failure.
# bool is explicitly rejected for numeric params: isinstance(True, int) is True
# in Python, so the check must come first.
# ---------------------------------------------------------------------------

_Validator = Callable[[Any], str | None]

_LOGREG_SOLVERS: frozenset[str] = frozenset(
    {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"}
)


def _positive_number(v: Any) -> str | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        return "must be a positive int or float"
    return None


def _positive_int(v: Any) -> str | None:
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        return "must be a positive int"
    return None


def _positive_int_or_none(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        return "must be a positive int or None"
    return None


def _int_or_none(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        return "must be an int or None"
    return None


def _logreg_class_weight(v: Any) -> str | None:
    if v not in ("balanced", None):
        return 'must be "balanced" or None'
    return None


def _logreg_solver(v: Any) -> str | None:
    if v not in _LOGREG_SOLVERS:
        return f"must be one of {sorted(_LOGREG_SOLVERS)}"
    return None


def _rf_class_weight(v: Any) -> str | None:
    if v not in ("balanced", "balanced_subsample", None):
        return 'must be "balanced", "balanced_subsample", or None'
    return None


def _min_samples_split(v: Any) -> str | None:
    """int >= 2 or float in (0, 1]."""
    if isinstance(v, bool):
        return "must be an int >= 2 or a float in (0, 1]"
    if isinstance(v, int) and v >= 2:
        return None
    if isinstance(v, float) and 0.0 < v <= 1.0:
        return None
    return "must be an int >= 2 or a float in (0, 1]"


def _min_samples_leaf(v: Any) -> str | None:
    """positive int (>= 1) or float in (0, 1) exclusive."""
    if isinstance(v, bool):
        return "must be a positive int or a float in (0, 1)"
    if isinstance(v, int) and v >= 1:
        return None
    if isinstance(v, float) and 0.0 < v < 1.0:
        return None
    return "must be a positive int or a float in (0, 1)"


_HYPERPARAMS_VALIDATORS: dict[ModelType, dict[str, _Validator]] = {
    ModelType.LOGREG: {
        "C": _positive_number,
        "max_iter": _positive_int,
        "random_state": _int_or_none,
        "class_weight": _logreg_class_weight,
        "solver": _logreg_solver,
    },
    ModelType.RANDOM_FOREST: {
        "n_estimators": _positive_int,
        "max_depth": _positive_int_or_none,
        "min_samples_split": _min_samples_split,
        "min_samples_leaf": _min_samples_leaf,
        "random_state": _int_or_none,
        "class_weight": _rf_class_weight,
    },
}


class TrainingConfigChurn(BaseModel):
    model_type: ModelType = ModelType.LOGREG
    hyperparameters: dict[str, int | float | str | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_hyperparameters(self) -> "TrainingConfigChurn":
        allowed = ALLOWED_HYPERPARAMS[self.model_type]
        unsupported = set(self.hyperparameters) - allowed
        if unsupported:
            raise ValueError(
                f"Unsupported hyperparameters for {self.model_type}: {sorted(unsupported)}. "
                f"Allowed: {sorted(allowed)}"
            )

        validators = _HYPERPARAMS_VALIDATORS[self.model_type]
        errors = [
            f"'{key}' {msg}"
            for key, value in self.hyperparameters.items()
            if key in validators and (msg := validators[key](value)) is not None
        ]
        if errors:
            raise ValueError(f"Invalid hyperparameter values: {'; '.join(errors)}")

        return self
