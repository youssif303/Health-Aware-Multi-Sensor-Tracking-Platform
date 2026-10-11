"""Estimate remaining useful life from health monitor features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rul_model import load_model, predict_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--sensor", choices=["camera", "radar"], required=True)
    parser.add_argument("--mean-nis", type=float, required=True)
    parser.add_argument("--bias-norm", type=float, required=True)
    parser.add_argument("--delivery-rate", type=float, default=1.0)
    parser.add_argument("--nis-trend-10", type=float, default=0.0)
    parser.add_argument("--bias-trend-10", type=float, default=0.0)
    args = parser.parse_args()
    model = load_model(args.model)
    features = [args.mean_nis, args.bias_norm, args.delivery_rate, args.nis_trend_10,
                args.bias_trend_10, float(args.sensor == "camera"), float(args.sensor == "radar")]
    prediction, uncertainty = predict_one(model, features)
    print(json.dumps({"sensor_id": args.sensor, "predicted_rul_seconds": prediction,
                      "neighbor_spread_seconds": uncertainty}, indent=2))


if __name__ == "__main__":
    main()
