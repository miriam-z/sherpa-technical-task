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

# Start main Streamlit UI (default port 8501, override with STREAMLIT_UI_PORT)
UI_PORT=${STREAMLIT_UI_PORT:-8501}
echo "[Entrypoint] Launching main Streamlit UI on http://localhost:$UI_PORT ..."
STREAMLIT_SERVER_PORT=$UI_PORT .venv/bin/streamlit run src/ui/app.py &

# Start TruLens dashboard (default port 8503, override with TRULENS_DASHBOARD_PORT)
DASHBOARD_PORT=${TRULENS_DASHBOARD_PORT:-8503}
echo "[Entrypoint] Launching TruLens dashboard/leaderboard on http://localhost:$DASHBOARD_PORT ..."
STREAMLIT_SERVER_PORT=$DASHBOARD_PORT PYTHONPATH=src python src/scripts/run_trulens_dashboard.py &

wait
