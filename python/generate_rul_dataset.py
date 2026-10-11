"""Generate labeled gradual-degradation histories for RUL model training."""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from pathlib import Path

from health_monitor import monitor
from simulate import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/rul_training"))
    parser.add_argument("--scenarios", type=int, default=40)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    if args.scenarios < 4 or args.steps < 120:
        parser.error("use at least 4 scenarios and 120 steps")
    if not args.tracker.is_file():
        parser.error(f"tracker executable does not exist: {args.tracker}")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    dataset_path = output / "rul_dataset.csv"
    rows: list[list[object]] = []
    for index in range(args.scenarios):
        fault = "camera_ramp" if index % 2 == 0 else "radar_ramp"
        sensor = "camera" if fault == "camera_ramp" else "radar"
        onset = rng.randint(max(10, args.steps // 5), args.steps * 2 // 5)
        failure = rng.randint(max(onset + 30, args.steps * 3 // 5), args.steps - 1)
        scenario = output / f"scenario_{index:03d}"
        generate(scenario, args.steps, args.seed + index, fault, onset, failure)
        tracker_path = scenario / "tracks.csv"
        innovation_path = scenario / "innovations.csv"
        subprocess.run([str(args.tracker.resolve()), str(scenario / "measurements.csv"),
                        str(tracker_path), str(innovation_path), "0.1"], check=True)
        monitor(scenario / "measurements.csv", innovation_path,
                scenario / "adaptive_measurements.csv", scenario / "health.csv")

        with (scenario / "health.csv").open(newline="", encoding="utf-8") as stream:
            health = [row for row in csv.DictReader(stream) if row["sensor_id"] == sensor]
        indexed = {int(row["step"]): row for row in health}
        for step in range(onset + 10, failure + 1, 5):
            current = indexed[step]
            previous = indexed.get(max(0, step - 10), current)
            rows.append([
                index, sensor, step,
                float(current["mean_nis"]), float(current["bias_norm"]),
                float(current["delivery_rate"]),
                float(current["mean_nis"]) - float(previous["mean_nis"]),
                float(current["bias_norm"]) - float(previous["bias_norm"]),
                (failure - step) * 0.1,
            ])
        print(f"Generated scenario {index + 1}/{args.scenarios}: {fault}, onset={onset}, failure={failure}")

    with dataset_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario_id", "sensor_id", "step", "mean_nis", "bias_norm",
                         "delivery_rate", "nis_trend_10", "bias_trend_10", "rul_seconds"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} labeled samples to {dataset_path}")


if __name__ == "__main__":
    main()
