"""Run matched baseline and health-adaptive replays for one scenario."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

from tracking_config import get_tracking_uri

from evaluate import evaluate
from simulate import generate
from health_monitor import monitor
from rul_model import load_model, predict_one


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def log_mlflow(result: dict[str, object], health: Path,
               rul_predictions: Path, rul_model: Path | None) -> tuple[str | None, str | None]:
    tracking_uri = get_tracking_uri()
    if not tracking_uri:
        return None, None
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "health-aware-tracking"))
        metrics = {
            "baseline_position_rmse": float(result["baseline"]["position_rmse"]),
            "adaptive_position_rmse": float(result["health_adaptive"]["position_rmse"]),
            "rmse_improvement_percent": float(result["rmse_improvement_percent"]),
            "track_availability": float(result["health_adaptive"]["track_availability"]),
            "false_alarm_rate": float(result["false_alarm_rate"]),
        }
        for key in ("fault_detection_delay_s", "rul_prediction_mae_seconds"):
            if result[key] is not None:
                metrics[key] = float(result[key])
        parameters = {key: result[key] for key in
                      ("seed", "steps", "fault", "fault_onset_step", "failure_threshold_step")
                      if result.get(key) is not None}
        with mlflow.start_run(run_name=f"{result['fault']}-seed-{result['seed']}") as active:
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"component": "tracking-platform", "adaptive_fusion": "enabled"})
            mlflow.log_artifact(str(health))
            if rul_predictions.is_file():
                mlflow.log_artifact(str(rul_predictions))
            if rul_model and rul_model.is_file():
                mlflow.log_artifact(str(rul_model), artifact_path="model")
            return active.info.run_id, None
    except Exception as error:  # Keep experiment output available when a tracking server is down.
        return None, f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True, help="path to compiled tracker executable")
    parser.add_argument("--output", type=Path, default=Path("data/experiment"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fault", choices=["none", "camera_noise", "radar_bias", "gps_dropout",
                                            "camera_ramp", "radar_ramp"], default="camera_noise")
    parser.add_argument("--onset-step", type=int)
    parser.add_argument("--failure-step", type=int)
    parser.add_argument("--rul-model", type=Path,
                        help="optional model path; used for camera_ramp or radar_ramp experiments")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if not args.tracker.is_file():
        parser.error(f"tracker executable does not exist: {args.tracker}")
    if args.rul_model and not args.rul_model.is_file():
        parser.error(f"RUL model does not exist: {args.rul_model}")
    if args.rul_model and args.fault not in {"camera_ramp", "radar_ramp"}:
        parser.error("RUL predictions are supported for gradual camera_ramp or radar_ramp scenarios")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    measurements = output / "measurements.csv"
    truth = output / "truth.csv"
    baseline_tracks = output / "baseline_tracks.csv"
    innovations = output / "innovations.csv"
    adjusted = output / "adaptive_measurements.csv"
    health = output / "health.csv"
    adaptive_tracks = output / "adaptive_tracks.csv"
    generate(output, args.steps, args.seed, args.fault, args.onset_step, args.failure_step)

    run([str(args.tracker), str(measurements), str(baseline_tracks), str(innovations), "0.1"])
    monitor(measurements, innovations, adjusted, health)
    run([str(args.tracker), str(adjusted), str(adaptive_tracks), str(output / "adaptive_innovations.csv"), "0.1"])

    baseline = evaluate(baseline_tracks, truth)
    adaptive = evaluate(adaptive_tracks, truth)
    improvement = 100.0 * (baseline["position_rmse"] - adaptive["position_rmse"]) / baseline["position_rmse"]
    expected_sensor = {"camera_noise": "camera", "camera_ramp": "camera",
                       "radar_bias": "radar", "radar_ramp": "radar",
                       "gps_dropout": "gps"}.get(args.fault)
    fault_start = args.steps // 2 if args.onset_step is None else args.onset_step
    with health.open(newline="", encoding="utf-8") as stream:
        health_rows = list(csv.DictReader(stream))
    expected_alerts = [int(row["step"]) for row in health_rows
                       if expected_sensor and row["sensor_id"] == expected_sensor
                       and int(row["step"]) >= fault_start and row["reason"] != "healthy"]
    excluded = lambda row: (expected_sensor is not None
                            and row["sensor_id"] == expected_sensor
                            and int(row["step"]) >= fault_start)
    false_alarm_rows = [row for row in health_rows if not excluded(row)]
    false_alarm_rate = (sum(row["reason"] != "healthy" for row in false_alarm_rows)
                        / len(false_alarm_rows)) if false_alarm_rows else 0.0
    rul_mae = None
    rul_sample_count = 0
    if args.rul_model:
        model = load_model(args.rul_model)
        sensor = expected_sensor
        sensor_rows = {int(row["step"]): row for row in health_rows if row["sensor_id"] == sensor}
        onset = fault_start
        threshold_step = args.steps - 1 if args.failure_step is None else args.failure_step
        prediction_rows = []
        errors = []
        for step in range(onset + 10, min(threshold_step, args.steps - 1) + 1):
            current = sensor_rows[step]
            previous = sensor_rows.get(max(0, step - 10), current)
            feature_values = [float(current["mean_nis"]), float(current["bias_norm"]),
                              float(current["delivery_rate"]),
                              float(current["mean_nis"]) - float(previous["mean_nis"]),
                              float(current["bias_norm"]) - float(previous["bias_norm"]),
                              float(sensor == "camera"), float(sensor == "radar")]
            prediction, spread = predict_one(model, feature_values)
            actual = (threshold_step - step) * 0.1
            errors.append(abs(prediction - actual))
            prediction_rows.append([step, sensor, f"{prediction:.6f}", f"{spread:.6f}", f"{actual:.6f}"])
        with (output / "rul_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "sensor_id", "predicted_rul_seconds", "neighbor_spread_seconds",
                             "actual_rul_seconds"])
            writer.writerows(prediction_rows)
        rul_sample_count = len(errors)
        rul_mae = sum(errors) / len(errors) if errors else None
    result = {
        "seed": args.seed,
        "steps": args.steps,
        "fault": args.fault,
        "fault_onset_step": fault_start if expected_sensor else None,
        "failure_threshold_step": (args.steps - 1 if args.failure_step is None else args.failure_step)
        if args.fault in {"camera_ramp", "radar_ramp"} else None,
        "baseline": baseline,
        "health_adaptive": adaptive,
        "rmse_improvement_percent": improvement,
        "fault_detection_delay_s": ((min(expected_alerts) - fault_start) * 0.1
                                     if expected_alerts else None),
        "false_alarm_rate": false_alarm_rate,
        "rul_prediction_mae_seconds": rul_mae,
        "rul_prediction_samples": rul_sample_count,
    }
    metrics_path = output / "metrics.json"
    rul_predictions = output / "rul_predictions.csv"
    mlflow_run_id, mlflow_error = log_mlflow(result, health, rul_predictions,
                                             args.rul_model)
    if get_tracking_uri():
        result["mlflow_run_id"] = mlflow_run_id
        result["mlflow_error"] = mlflow_error
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved matched-run metrics to {metrics_path}")


if __name__ == "__main__":
    main()
