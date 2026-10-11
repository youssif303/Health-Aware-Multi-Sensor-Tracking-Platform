"""Generate a deterministic three-target, three-sensor tracking scenario."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

DT = 0.1
SENSORS = {"radar": 1.8, "camera": 0.9, "gps": 0.5}


def position(target: int, t: float) -> tuple[float, float]:
    phase = target * 2.0 * math.pi / 3.0
    return (30.0 * math.cos(0.12 * t + phase) + target * 12.0,
            22.0 * math.sin(0.12 * t + phase) + target * 8.0)


def generate(output: Path, steps: int, seed: int, fault: str,
             onset_step: int | None = None, failure_step: int | None = None) -> None:
    rng = random.Random(seed)
    onset_step = steps // 2 if onset_step is None else onset_step
    failure_step = steps - 1 if failure_step is None else failure_step
    if failure_step <= onset_step:
        raise ValueError("failure_step must be greater than onset_step")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as measurements_file, \
         (output / "truth.csv").open("w", newline="", encoding="utf-8") as truth_file:
        measurements = csv.writer(measurements_file)
        truth = csv.writer(truth_file)
        measurements.writerow(["step", "target_id", "sensor_id", "x", "y", "sigma"])
        truth.writerow(["step", "time_s", "target_id", "x", "y"])
        for step in range(steps):
            t = step * DT
            for target in range(3):
                x, y = position(target, t)
                truth.writerow([step, f"{t:.3f}", target, f"{x:.6f}", f"{y:.6f}"])
                for sensor, base_sigma in SENSORS.items():
                    sigma = base_sigma
                    bias_x = bias_y = 0.0
                    progress = min(1.0, max(0.0, (step - onset_step) / (failure_step - onset_step)))
                    if fault == "camera_noise" and sensor == "camera" and step >= steps // 2:
                        sigma *= 5.0
                    elif fault == "radar_bias" and sensor == "radar" and step >= steps // 2:
                        bias_x = 12.0
                    elif fault == "gps_dropout" and sensor == "gps" and step >= steps // 2:
                        if step % 4 != 0:
                            continue
                    elif fault == "camera_ramp" and sensor == "camera":
                        sigma *= 1.0 + 4.0 * progress
                    elif fault == "radar_ramp" and sensor == "radar":
                        bias_x = 10.0 * progress
                    # The tracker receives the nominal sensor covariance. Fault severity is
                    # hidden from it so later health logic must infer degradation from data.
                    mx = x + bias_x + rng.gauss(0.0, sigma)
                    my = y + bias_y + rng.gauss(0.0, sigma)
                    measurements.writerow([step, target, sensor, f"{mx:.6f}", f"{my:.6f}", f"{base_sigma:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/run"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fault", choices=["none", "camera_noise", "radar_bias", "gps_dropout",
                                            "camera_ramp", "radar_ramp"], default="none")
    parser.add_argument("--onset-step", type=int)
    parser.add_argument("--failure-step", type=int)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    generate(args.output, args.steps, args.seed, args.fault, args.onset_step, args.failure_step)
    print(f"Wrote {args.output / 'measurements.csv'} and {args.output / 'truth.csv'}")


if __name__ == "__main__":
    main()
