# Multi-purpose Dockerfile for a Python AI/ML web service
# - Installs common build deps (for wheels, psycopg2, cryptography, etc.)
# - Runs as non-root user
# - Default runtime tries common entrypoints; override with docker run or docker-compose

FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Install system deps for building common Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential gcc curl ca-certificates git \
       libpq-dev libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app
COPY . /app

# Install Python dependencies:
# - If requirements.txt exists, use pip
# - Else if pyproject.toml exists, install Poetry and use it to install deps
RUN set -eux; \
    if [ -f "requirements.txt" ]; then \
        pip install --upgrade pip setuptools wheel; \
        pip install -r requirements.txt; \
    elif [ -f "pyproject.toml" ]; then \
        pip install --upgrade pip; \
        pip install poetry; \
        poetry config virtualenvs.create false --local || true; \
        poetry install --no-interaction --no-ansi --no-root || true; \
    else \
        echo "No requirements.txt or pyproject.toml found, skipping dependency installation"; \
    fi

# Train model during build (Option A) - ensures model matches runtime library versions
# Requires data/raw/water_data_quality.csv in repo. Migrate from LFS: git lfs migrate export --include="data/*,models/*" --everything
RUN python run_pipeline.py train

# Ensure app files owned by non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE ${PORT}

# Run Hydronew AI classification service
ENTRYPOINT ["sh", "-c"]
CMD ["python run_pipeline.py classify"]