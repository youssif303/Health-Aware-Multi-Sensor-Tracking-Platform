"""Infer sensor health from innovation residuals and create an adaptive replay."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def monitor(measurements_path: Path, innovations_path: Path,
            adjusted_path: Path, health_path: Path, window_steps: int = 30) -> None:
    measurements = read_csv(measurements_path)
    innovations = read_csv(innovations_path)
    if not measurements:
        raise ValueError("measurement file is empty")
    sensors = sorted({row["sensor_id"] for row in measurements})
    steps = sorted({int(row["step"]) for row in measurements})
    target_count = len({row["target_id"] for row in measurements})
    if target_count == 0:
        raise ValueError("no targets found")

    innovations_by_step_sensor: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    detections_by_step_sensor: dict[tuple[int, str], int] = defaultdict(int)
    for row in measurements:
        detections_by_step_sensor[(int(row["step"]), row["sensor_id"])] += 1
    for row in innovations:
        sx, sy = float(row["sxx"]), float(row["syy"])
        if sx <= 0 or sy <= 0:
            continue
        innovations_by_step_sensor[(int(row["step"]), row["sensor_id"])].append(
            (float(row["dx"]) / math.sqrt(sx), float(row["dy"]) / math.sqrt(sy))
        )

    residual_history = {sensor: deque(maxlen=window_steps * target_count) for sensor in sensors}
    delivery_history = {sensor: deque(maxlen=window_steps) for sensor in sensors}
    multiplier_by_step: dict[tuple[int, str], float] = {}
    health_rows: list[list[object]] = []
    for step in range(min(steps), max(steps) + 1):
        for sensor in sensors:
            current = innovations_by_step_sensor.get((step, sensor), [])
            residual_history[sensor].extend(current)
            delivery_history[sensor].append(
                min(1.0, detections_by_step_sensor.get((step, sensor), 0) / target_count)
            )

            samples = residual_history[sensor]
            if samples:
                mean_nis = sum(x * x + y * y for x, y in samples) / len(samples)
                mean_x = sum(x for x, _ in samples) / len(samples)
                mean_y = sum(y for _, y in samples) / len(samples)
                bias_norm = math.hypot(mean_x, mean_y)
            else:
                mean_nis, bias_norm = 2.0, 0.0
            delivery = sum(delivery_history[sensor]) / len(delivery_history[sensor])

            noise_factor = max(1.0, mean_nis / 2.0)
            bias_factor = 1.0 + bias_norm * bias_norm
            # Missing packets are already skipped by the tracker. Report the outage in
            # health, but do not penalize the covariance of the accurate packets that arrive.
            multiplier = min(100.0, max(1.0, noise_factor, bias_factor))
            degradation_score = max(1.0 - 1.0 / multiplier, 1.0 - delivery)
            reasons = []
            if mean_nis > 4.0:
                reasons.append("elevated_innovation_variance")
            if bias_norm > 1.5:
                reasons.append("persistent_innovation_bias")
            if delivery < 0.95:
                reasons.append("missing_measurements")
            multiplier_by_step[(step, sensor)] = multiplier
            health_rows.append([step, sensor, f"{mean_nis:.6f}", f"{bias_norm:.6f}",
                                f"{delivery:.6f}", f"{degradation_score:.6f}",
                                f"{multiplier:.6f}", ";".join(reasons) or "healthy"])

    adjusted_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.parent.mkdir(parents=True, exist_ok=True)
    with adjusted_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "target_id", "sensor_id", "x", "y", "sigma"])
        for row in measurements:
            step, sensor = int(row["step"]), row["sensor_id"]
            factor = multiplier_by_step[(step, sensor)]
            writer.writerow([row["step"], row["target_id"], sensor, row["x"], row["y"],
                             f"{float(row['sigma']) * math.sqrt(factor):.6f}"])
    with health_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "sensor_id", "mean_nis", "bias_norm", "delivery_rate",
                         "degradation_score", "covariance_multiplier", "reason"])
        writer.writerows(health_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("innovations", type=Path)
    parser.add_argument("adjusted", type=Path)
    parser.add_argument("health", type=Path)
    parser.add_argument("--window-steps", type=int, default=30)
    args = parser.parse_args()
    if args.window_steps < 2:
        parser.error("--window-steps must be at least 2")
    monitor(args.measurements, args.innovations, args.adjusted, args.health, args.window_steps)
    print(f"Wrote adaptive measurements to {args.adjusted} and health history to {args.health}")


if __name__ == "__main__":
    main()
