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

## Project Structure

```
src/churn_service/
├── main.py              — app factory
├── core/config.py       — settings via environment variables
├── schemas/features.py  — Pydantic input models
├── services/            — business logic (ML model)
└── api/v1/              — HTTP endpoints

data/
└── churn_dataset.csv    — training dataset

tests/
├── unit/                — schema validation tests
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
