#!/bin/sh
set -eu

mkdir -p /tmp/mlflow /app/data
api_pid=
mlflow_pid=

cleanup() {
    if [ -n "$api_pid" ]; then kill "$api_pid" 2>/dev/null || true; fi
    if [ -n "$mlflow_pid" ]; then kill "$mlflow_pid" 2>/dev/null || true; fi
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

mlflow server \
    --backend-store-uri sqlite:////tmp/mlflow/mlflow.db \
    --artifacts-destination /tmp/mlflow/artifacts \
    --host 127.0.0.1 \
    --port 5000 \
    --workers 1 \
    --allowed-hosts localhost,127.0.0.1,localhost:5000,127.0.0.1:5000 &
mlflow_pid=$!

python - <<'PY'
import socket
import time

for _ in range(120):
    try:
        with socket.create_connection(("127.0.0.1", 5000), timeout=1):
            pass
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("MLflow did not become ready")
PY

uvicorn api:app --app-dir /app/python --host 0.0.0.0 --port "${PORT:-10000}" &
api_pid=$!
wait "$api_pid"
