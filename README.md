# Churn Service

FastAPI service for customer churn prediction using scikit-learn.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Quick Start

```bash
make install
make run
```

API available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

## Commands

| Command | Description |
|---|---|
| `make install` | Install dependencies |
| `make run` | Start development server with auto-reload |
| `make test` | Run tests |
| `make lint` | Check code style |
| `make lint-fix` | Auto-fix code style |

## Configuration

Copy `.env.example` to `.env` and adjust values if needed:

```bash
cp .env.example .env
```

Key settings: `DATASET_PATH` (default: `data/churn_dataset.csv`), `MODELS_DIR` (default: `models/`).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/` | Health check |
| GET | `/api/v1/dataset/info` | Dataset statistics and class distribution |
| GET | `/api/v1/dataset/split-info` | Train/test split sizes and feature lists |
| GET | `/api/v1/dataset/preview` | First N rows of the dataset |
| POST | `/api/v1/model/train` | Train and persist the churn model |
| GET | `/api/v1/model/status` | Model status, training timestamp, and metrics |
| POST | `/api/v1/predict/` | Predict churn for one or more customers |

## Usage

```bash
# Train the model
curl -X POST http://localhost:8000/api/v1/model/train

# Check model status and metrics
curl http://localhost:8000/api/v1/model/status

# Predict churn for one or more customers
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "monthly_fee": 29.99,
        "usage_hours": 120,
        "support_requests": 3,
        "account_age_months": 24,
        "failed_payments": 0,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "card",
        "autopay_enabled": 1
      }
    ]
  }'

# Response:
# {
#   "predictions": [
#     {
#       "predicted_class": 0,
#       "churn_probability": 0.08,
#       "retained_probability": 0.92
#     }
#   ]
# }
```

The trained model is saved to `models/churn_model.joblib` and loaded automatically on the next restart.

## Project Structure

```
src/churn_service/
├── main.py              — app factory with startup lifespan
├── core/                — settings, exceptions
├── schemas/             — Pydantic request/response models
├── services/            — dataset, preprocessing, training, model storage
└── api/v1/              — HTTP endpoints

data/
└── churn_dataset.csv    — training dataset

models/                  — persisted model files (gitignored)

tests/
├── unit/                — service and schema tests
└── integration/         — HTTP endpoint tests
```

## Feature Schema

| Feature | Type | Description |
|---|---|---|
| `monthly_fee` | float | Monthly subscription fee in USD |
| `usage_hours` | float | Hours of service usage in the past month |
| `support_requests` | int | Number of support tickets submitted |
| `account_age_months` | int | Account age in months |
| `failed_payments` | int | Number of failed payment attempts |
| `region` | enum | `europe` / `asia` / `america` / `africa` |
| `device_type` | enum | `mobile` / `desktop` / `tablet` |
| `payment_method` | enum | `card` / `paypal` / `crypto` |
| `autopay_enabled` | 0 or 1 | Whether autopay is enabled |
