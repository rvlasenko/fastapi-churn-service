import typing
from typing import Any

from churn_service.schemas.features import CATEGORICAL_FEATURE_ENUMS, FeatureVectorChurn
from churn_service.schemas.model import (
    CategoricalFeatureSchema,
    ModelSchemaResponse,
    NumericalFeatureSchema,
)
from churn_service.services.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN


def _resolve_field_type(annotation: Any) -> str:
    """Map a Pydantic field annotation to a schema type string.

    Unwraps Annotated[T, ...] before checking the base type.
    Raises ValueError for unrecognised types so contract bugs surface immediately.
    """
    if typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    if annotation is float:
        return "float"
    if annotation is int:
        return "int"
    if annotation is bool:
        return "bool"
    if isinstance(annotation, type) and issubclass(annotation, str):
        return "str"
    raise ValueError(f"Unsupported field annotation for schema: {annotation}")


def build_model_schema() -> ModelSchemaResponse:
    """Build the static model input schema from constants and FeatureVectorChurn."""
    fields = FeatureVectorChurn.model_fields
    return ModelSchemaResponse(
        numerical_features=[
            NumericalFeatureSchema(
                name=name,
                type=_resolve_field_type(fields[name].annotation),
            )
            for name in NUMERICAL_FEATURES
        ],
        categorical_features=[
            CategoricalFeatureSchema(
                name=name,
                type="str",
                accepted_values=[v.value for v in CATEGORICAL_FEATURE_ENUMS[name]],
            )
            for name in CATEGORICAL_FEATURES
        ],
        target=TARGET_COLUMN,
    )
