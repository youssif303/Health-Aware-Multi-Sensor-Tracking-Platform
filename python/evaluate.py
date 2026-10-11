"""Compare estimated positions with simulator ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def evaluate(tracks_path: Path, truth_path: Path) -> dict[str, float | int]:
    estimates = {(int(r["step"]), int(r["target_id"])): r for r in read_rows(tracks_path)}
    truth = read_rows(truth_path)
    squared_error = 0.0
    count = 0
    for row in truth:
        key = (int(row["step"]), int(row["target_id"]))
        estimate = estimates.get(key)
        if estimate is None:
            continue
        dx = float(estimate["x"]) - float(row["x"])
        dy = float(estimate["y"]) - float(row["y"])
        squared_error += dx * dx + dy * dy
        count += 1
    if count == 0:
        raise ValueError("no overlapping track and truth rows")
    return {
        "position_rmse": math.sqrt(squared_error / count),
        "tracked_target_steps": count,
        "truth_target_steps": len(truth),
        "track_availability": count / len(truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    metrics = evaluate(args.tracks, args.truth)
    rendered = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
