# =============================================================================
# RailPulse — Railway Cascade Prediction dashboard (FastAPI + Leaflet/OSM)
# Render-compatible image. Binds 0.0.0.0 on $PORT (Render injects it).
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Native libs: libgomp1 = OpenMP runtime for LightGBM/XGBoost/scikit-learn,
# freetype/png for matplotlib (only needed if the model retrains on the box).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libfreetype6 libpng16-16 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencies first (kept as a separate layer so rebuilds are fast).
#    deploy/requirements.txt is the lean runtime set (no Gradio; CatBoost is
#    included so the persisted model files all load on the deployed box).
COPY deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) Application code + assets.
#    config.py computes ROOT = parent of src/ = /app, which matches this layout
#    exactly, so /data, /results, /models and /lib all resolve correctly.
COPY src      ./src
COPY lib      ./lib
COPY data     ./data
COPY results  ./results
COPY models   ./models

WORKDIR /app/src

EXPOSE 8000

# Render sets $PORT; default to 8000 for local `docker run`.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
