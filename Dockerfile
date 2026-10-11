FROM python:3.12-slim-bookworm AS cpp-build
RUN apt-get update && apt-get install -y --no-install-recommends cmake g++ make && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY cpp/ ./cpp/
RUN cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release

FROM python:3.12-slim-bookworm AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRACKER_EXECUTABLE=/usr/local/bin/tracker \
    RUL_MODEL_PATH=/app/models/rul_knn.json \
    MLFLOW_TRACKING_URI=http://mlflow:5000
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/data /mlflow && chown -R app:app /app /mlflow
WORKDIR /app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod 0755 /usr/local/bin/docker-entrypoint
COPY requirements-tracking.txt requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements-tracking.txt
COPY --chown=app:app python/ ./python/
COPY --chown=app:app dashboard/ ./dashboard/
COPY --chown=app:app models/ ./models/
COPY --chown=app:app results/ ./results/
COPY --chown=app:app render-free-start.sh /usr/local/bin/render-free-start
RUN chmod 0755 /usr/local/bin/render-free-start
COPY --from=cpp-build /src/build/tracker /usr/local/bin/tracker
ENTRYPOINT ["/usr/local/bin/docker-entrypoint"]
EXPOSE 8000 5000
CMD ["sh", "-c", "exec uvicorn api:app --app-dir python --host 0.0.0.0 --port ${PORT:-8000}"]
