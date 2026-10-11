"""Resolve the MLflow URI for local Compose and Render private networking."""

from __future__ import annotations

import os


def get_tracking_uri() -> str | None:
    """Use an explicit URI, or construct one from Render's private host:port."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    hostport = os.environ.get("MLFLOW_TRACKING_HOSTPORT")
    return f"http://{hostport}" if hostport else None
