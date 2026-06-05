FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

WORKDIR /app

# Layer 1: third-party deps (cached unless pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# Layer 2: install the local package
COPY src/ ./src/
RUN uv sync --no-dev --frozen

# Layer 3: dataset (baked into image)
COPY data/ ./data/

# Layer 4: writable models directory — no content, written at runtime
RUN mkdir -p models

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "churn_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
