FROM python:3.11-slim

WORKDIR /app

# Copy repo (run `git lfs pull` on the host before building so real files are in the context)
COPY . .

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "run_pipeline.py", "classify"]
