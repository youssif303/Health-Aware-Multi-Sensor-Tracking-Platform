"""Run matched tracking experiments over several seeds and aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

from tracking_config import get_tracking_uri


METRICS = ["baseline_position_rmse", "adaptive_position_rmse", "rmse_improvement_percent",
           "fault_detection_delay_s", "false_alarm_rate", "rul_prediction_mae_seconds"]
BOOTSTRAP_RESAMPLES = 10_000


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "stddev": None, "median": None,
                "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    ci_low = ci_high = None
    if len(values) > 1:
        rng = random.Random(0)
        bootstrap_means = [
            statistics.fmean(values[rng.randrange(len(values))] for _ in values)
            for _ in range(BOOTSTRAP_RESAMPLES)
        ]
        bootstrap_means.sort()
        ci_low = bootstrap_means[int(0.025 * BOOTSTRAP_RESAMPLES)]
        ci_high = bootstrap_means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return {
        "count": len(values),
        "mean": mean,
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/sweep"))
    parser.add_argument("--faults", nargs="+", default=["camera_noise", "radar_bias", "gps_dropout", "camera_ramp", "radar_ramp"])
    parser.add_argument("--seed-start", type=int, default=7)
    parser.add_argument("--runs-per-fault", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--rul-model", type=Path, default=Path("models/rul_knn.json"))
    args = parser.parse_args()
    known = {"none", "camera_noise", "radar_bias", "gps_dropout", "camera_ramp", "radar_ramp"}
    if not set(args.faults) <= known:
        parser.error(f"faults must be selected from: {', '.join(sorted(known))}")
    if args.runs_per_fault < 2 or args.steps < 2:
        parser.error("use at least two runs per fault and two steps")
    if not args.tracker.is_file():
        parser.error(f"tracker executable does not exist: {args.tracker}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("run_experiment.py")
    rows: list[dict[str, object]] = []
    for fault in args.faults:
        for offset in range(args.runs_per_fault):
            seed = args.seed_start + offset
            run_output = output / fault / f"seed_{seed}"
            command = [sys.executable, str(script), "--tracker", str(args.tracker.resolve()),
                       "--output", str(run_output.resolve()), "--steps", str(args.steps),
                       "--seed", str(seed), "--fault", fault]
            if fault in {"camera_ramp", "radar_ramp"} and args.rul_model.is_file():
                command.extend(["--rul-model", str(args.rul_model.resolve())])
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1],
                           stdout=subprocess.DEVNULL)
            result = json.loads((run_output / "metrics.json").read_text(encoding="utf-8"))
            rows.append({
                "fault": fault,
                "baseline_position_rmse": result["baseline"]["position_rmse"],
                "adaptive_position_rmse": result["health_adaptive"]["position_rmse"],
                **{key: result.get(key) for key in METRICS if key not in {
                    "baseline_position_rmse", "adaptive_position_rmse"
                }},
                "seed": seed,
            })
            print(f"Completed {fault}, seed {seed}")

    summary: dict[str, object] = {
        "runs_per_fault": args.runs_per_fault,
        "steps": args.steps,
        "seed_start": args.seed_start,
        "confidence_interval": {"method": "percentile_bootstrap", "level": 0.95,
                                "resamples": BOOTSTRAP_RESAMPLES,
                                "unit": "paired per-seed RMSE improvement"},
        "faults": {},
    }
    for fault in args.faults:
        fault_rows = [row for row in rows if row["fault"] == fault]
        summary["faults"][fault] = {
            metric: summarize([float(row[metric]) for row in fault_rows if row[metric] is not None])
            for metric in METRICS
        }
    summary_path = output / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows_path = output / "sweep_runs.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fault", "seed", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    tracking_uri = get_tracking_uri()
    if tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "health-aware-tracking"))
            with mlflow.start_run(run_name="multi-seed-sweep", tags={"component": "sweep"}):
                mlflow.log_params({"runs_per_fault": args.runs_per_fault, "steps": args.steps,
                                   "seed_start": args.seed_start, "faults": ",".join(args.faults)})
                aggregate_metrics = {}
                for fault, values in summary["faults"].items():
                    improvement = values["rmse_improvement_percent"]["mean"]
                    if improvement is not None:
                        aggregate_metrics[f"{fault}_mean_rmse_improvement_percent"] = improvement
                mlflow.log_metrics(aggregate_metrics)
                mlflow.log_artifact(str(summary_path))
                mlflow.log_artifact(str(rows_path))
        except Exception as error:
            print(f"MLflow sweep logging failed: {type(error).__name__}: {error}", file=sys.stderr)
    print(json.dumps(summary, indent=2))
    print(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
