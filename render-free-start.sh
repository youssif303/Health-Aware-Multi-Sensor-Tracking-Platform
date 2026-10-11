#!/bin/sh
set -eu

mkdir -p /app/data

# Render's free 512 MiB instance cannot run MLflow and the API together.
# Keep tracking-server integration enabled in Docker Compose, but disable it
# in this single-process free demo so the public API has memory to run.
unset MLFLOW_TRACKING_URI

exec uvicorn api:app \
    --app-dir /app/python \
    --host 0.0.0.0 \
    --port "${PORT:-10000}"
