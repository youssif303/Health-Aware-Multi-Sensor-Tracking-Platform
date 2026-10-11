"""Shared inference helpers for the saved k-nearest-neighbors RUL model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_model(path: Path) -> dict:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("model_type") != "knn_regressor":
        raise ValueError("unsupported RUL model type")
    return model


def predict_one(model: dict, values: list[float]) -> tuple[float, float]:
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    query = (np.asarray(values, dtype=float) - mean) / scale
    support = np.asarray(model["support_vectors"])
    rul = np.asarray(model["remaining_seconds"])
    distance = np.sum((support - query) ** 2, axis=1)
    k = min(int(model["neighbors"]), len(support))
    indices = np.argpartition(distance, k - 1)[:k]
    weights = 1.0 / np.maximum(np.sqrt(distance[indices]), 1e-6)
    values = rul[indices]
    prediction = float(np.sum(values * weights) / np.sum(weights))
    spread = float(np.sum(weights * np.abs(values - prediction)) / np.sum(weights))
    return max(0.0, prediction), spread
