from enum import StrEnum

from pydantic import BaseModel, Field


class Region(StrEnum):
    EUROPE = "europe"
    ASIA = "asia"
    AMERICA = "america"
    AFRICA = "africa"


class DeviceType(StrEnum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"


class PaymentMethod(StrEnum):
    CARD = "card"
    PAYPAL = "paypal"
    CRYPTO = "crypto"


CATEGORICAL_FEATURE_ENUMS: dict[str, type[StrEnum]] = {
    "region": Region,
    "device_type": DeviceType,
    "payment_method": PaymentMethod,
}


class FeatureVectorChurn(BaseModel):
    monthly_fee: float = Field(..., gt=0, le=10_000, description="Monthly subscription fee in USD")
    usage_hours: float = Field(
        ..., ge=0, le=744, description="Hours of service usage in the past month (max 31×24)"
    )
    support_requests: int = Field(
        ..., ge=0, le=1_000, description="Number of support tickets submitted"
    )
    account_age_months: int = Field(..., ge=0, le=600, description="Account age in months")
    failed_payments: int = Field(..., ge=0, le=100, description="Number of failed payment attempts")
    region: Region
    device_type: DeviceType
    payment_method: PaymentMethod
    autopay_enabled: int = Field(..., ge=0, le=1, description="Whether autopay is enabled (0 or 1)")


class DatasetRowChurn(FeatureVectorChurn):
    churn: int = Field(..., ge=0, le=1, description="Ground truth label: 1 = churned, 0 = retained")
