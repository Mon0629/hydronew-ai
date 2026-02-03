FROM python:3.11-slim

# Install git + git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy repo
COPY . .

# Pull LFS objects (this replaces the 131-byte pointer with the real file)
RUN git lfs pull

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "run_pipeline.py", "classify"]
