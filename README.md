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
| `make docker-build` | Build Docker image |
| `make docker-run` | Run container on port 8000 |

## Docker

Build the image:

```bash
make docker-build
```

Run the container (models are ephemeral — lost when the container stops):

```bash
make docker-run
```

Run with a persistent models volume (trained model survives container restarts):

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" churn-service
```

Service available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

> `models/` is runtime data. The image ships without a trained model.
> Train after startup via `POST /api/v1/model/train`.
> Without a volume mount, model files written inside the container are lost when the container stops.

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
| POST | `/api/v1/model/train` | Train and persist the churn model (optional config body) |
| GET | `/api/v1/model/status` | Model status, training timestamp, metrics, and model config |
| GET | `/api/v1/model/schema` | Input field names, types, and allowed enum values |
| GET | `/api/v1/model/metrics` | Training history (supports `?model_type=` and `?limit=` filters) |
| POST | `/api/v1/predict/` | Predict churn for one or more customers |

## Usage

```bash
# Train with default model (LogisticRegression)
curl -X POST http://localhost:8000/api/v1/model/train

# Train with RandomForestClassifier and custom hyperparameters
curl -X POST http://localhost:8000/api/v1/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 10}}'

# Train LogisticRegression with custom C
curl -X POST http://localhost:8000/api/v1/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {"C": 0.5}}'

# Check model status, metrics, and training config
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

### Supported models

| `model_type` | Algorithm | Allowed hyperparameters |
|---|---|---|
| `logreg` | LogisticRegression | `C`, `max_iter`, `class_weight`, `solver`, `random_state` |
| `random_forest` | RandomForestClassifier | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`, `random_state` |

Default values: `random_state=42`, `max_iter=1000` (logreg only). Unsupported hyperparameter names return `422`.

## Project Structure

```
src/churn_service/
├── main.py              — app factory with startup lifespan
├── core/                — settings, exceptions
├── schemas/             — Pydantic request/response models
├── services/            — dataset, preprocessing, pipeline builder, training, model storage
└── api/v1/              — HTTP endpoints

data/
└── churn_dataset.csv    — training dataset

models/                  — persisted model files (gitignored)

tests/
├── unit/                — service and schema tests
└── integration/         — HTTP endpoint tests
```

## Error Responses

All errors use a common JSON format:

```json
{
  "error": {
    "code": "stable_error_code",
    "message": "Human-readable message",
    "details": null
  }
}
```

**Invalid `/model/train` hyperparameter → 422**
```bash
curl -X POST http://localhost:8000/api/v1/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": -1}}'
# {"error": {"code": "validation_error", "message": "Request validation failed", "details": [...]}}
```

**`/predict` without trained model → 503**
```bash
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"items": [...]}'
# {"error": {"code": "model_not_trained", "message": "No trained model available...", "details": null}}
```

**Invalid field in `/predict` → 422**
```bash
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"items": [{"region": "narnia", ...}]}'
# {"error": {"code": "validation_error", "message": "Request validation failed", "details": [...]}}
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
