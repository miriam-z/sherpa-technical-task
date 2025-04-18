#!/bin/bash
set -e

# Create venv with uv if not exists
if [ ! -d ".venv" ]; then
  uv venv --python 3.12
fi

# Activate venv
source .venv/bin/activate

# Source .env file if present
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Install dependencies with uv
uv sync

# Run pipeline if requested
if [ "$RUN_PIPELINE_ON_START" = "true" ]; then
  echo "[Entrypoint] Running document processing pipeline..."
  PYTHONPATH=src python src/main.py
fi

# Start FastAPI backend
PYTHONPATH=src .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit UI
.venv/bin/streamlit run src/ui/app.py --server.port 8501

wait
