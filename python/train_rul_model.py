"""Train and evaluate a small k-nearest-neighbors RUL regressor."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

from tracking_config import get_tracking_uri

import numpy as np

FEATURES = ["mean_nis", "bias_norm", "delivery_rate", "nis_trend_10",
            "bias_trend_10", "sensor_is_camera", "sensor_is_radar"]


def encode(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, groups = [], [], []
    for row in rows:
        sensor = row["sensor_id"]
        features.append([
            float(row["mean_nis"]), float(row["bias_norm"]), float(row["delivery_rate"]),
            float(row["nis_trend_10"]), float(row["bias_trend_10"]),
            float(sensor == "camera"), float(sensor == "radar"),
        ])
        labels.append(float(row["rul_seconds"]))
        groups.append(int(row["scenario_id"]))
    return np.asarray(features), np.asarray(labels), np.asarray(groups)


def predict(query: np.ndarray, reference: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    predictions = np.empty(query.shape[0], dtype=float)
    for start in range(0, query.shape[0], 128):
        batch = query[start:start + 128]
        distances = np.sum((batch[:, None, :] - reference[None, :, :]) ** 2, axis=2)
        nearest = np.argpartition(distances, min(k - 1, reference.shape[0] - 1), axis=1)[:, :k]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        weights = 1.0 / np.maximum(np.sqrt(nearest_distances), 1e-6)
        values = labels[nearest]
        predictions[start:start + batch.shape[0]] = np.sum(values * weights, axis=1) / np.sum(weights, axis=1)
    return np.maximum(predictions, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/rul_knn.json"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--neighbors", type=int, default=15)
    args = parser.parse_args()
    if not 0.1 <= args.test_fraction <= 0.5:
        parser.error("--test-fraction must be between 0.1 and 0.5")
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")
    with args.dataset.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 10:
        parser.error("dataset must contain at least 10 rows")
    x, y, groups = encode(rows)
    scenarios = sorted(set(groups.tolist()))
    random.Random(args.seed).shuffle(scenarios)
    test_count = max(1, round(len(scenarios) * args.test_fraction))
    test_scenarios = set(scenarios[:test_count])
    train_mask = np.asarray([group not in test_scenarios for group in groups])
    test_mask = ~train_mask
    mean = x[train_mask].mean(axis=0)
    scale = x[train_mask].std(axis=0)
    scale[scale < 1e-9] = 1.0
    train_x = (x[train_mask] - mean) / scale
    test_x = (x[test_mask] - mean) / scale
    k = min(args.neighbors, int(train_mask.sum()))
    predicted = predict(test_x, train_x, y[train_mask], k)
    actual = y[test_mask]
    mae = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    baseline_mae = float(np.mean(np.abs(actual - np.median(y[train_mask]))))

    model = {
        "model_type": "knn_regressor",
        "target": "remaining_seconds_until_simulated_failure_threshold",
        "features": FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "support_vectors": train_x.tolist(),
        "remaining_seconds": y[train_mask].tolist(),
        "neighbors": k,
        "training_scenarios": sorted(set(groups[train_mask].tolist())),
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    metrics = {
        "training_samples": int(train_mask.sum()),
        "held_out_samples": int(test_mask.sum()),
        "training_scenarios": int(len(scenarios) - test_count),
        "held_out_scenarios": int(test_count),
        "mae_seconds": mae,
        "rmse_seconds": rmse,
        "median_baseline_mae_seconds": baseline_mae,
        "mae_improvement_percent": 100.0 * (baseline_mae - mae) / baseline_mae if baseline_mae else 0.0,
        "model_path": str(args.model),
    }
    metrics_path = args.model.with_name("rul_model_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    tracking_uri = get_tracking_uri()
    if tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "health-aware-tracking"))
            with mlflow.start_run(run_name="rul-knn-training", tags={"component": "rul-training"}):
                mlflow.log_params({
                    "seed": args.seed,
                    "test_fraction": args.test_fraction,
                    "neighbors": k,
                    "dataset_rows": len(rows),
                    "features": ",".join(FEATURES),
                })
                mlflow.log_metrics({key: float(metrics[key]) for key in (
                    "mae_seconds", "rmse_seconds", "median_baseline_mae_seconds",
                    "mae_improvement_percent",
                )})
                mlflow.log_artifact(str(args.dataset.resolve()))
                mlflow.log_artifact(str(args.model.resolve()))
                mlflow.log_artifact(str(metrics_path.resolve()))
        except Exception as error:
            import sys
            print(f"MLflow RUL training logging failed: {type(error).__name__}: {error}", file=sys.stderr)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
