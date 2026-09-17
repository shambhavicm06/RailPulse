#!/usr/bin/env bash
# One-command launcher for the RailPulse Railway Cascade Predictor.
# Starts the FastAPI backend (8000) + Gradio dashboard (7860).
set -e
cd "$(dirname "$0")"

# Prefer the project venv if it exists, else the system python.
PY="python"
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi

if [ ! -f models/bundle.joblib ]; then
  echo "No trained model found. Running the full pipeline first ..."
  echo "(generate data -> features -> train 10 models)"
  "$PY" src/data_generator.py
  "$PY" src/feature_engineering.py
  "$PY" src/train.py
fi

echo
echo "Starting FastAPI backend (8000) + Gradio dashboard (7860) ..."
echo "  RailPulse web app  (login + dark/light + network map): http://127.0.0.1:8000"
echo "  Gradio dispatcher console:                             http://127.0.0.1:7860"
echo "  Login -> username: admin   password: swr2026"
echo
"$PY" src/run.py
