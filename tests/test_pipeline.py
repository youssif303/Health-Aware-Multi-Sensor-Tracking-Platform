import csv
import os
import subprocess
from pathlib import Path

import pytest

import api
from evaluate import evaluate
from health_monitor import monitor
from rul_model import predict_one
from simulate import generate
from sweep_experiments import summarize


ROOT = Path(__file__).resolve().parents[1]


def test_simulator_is_reproducible(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    generate(first, steps=30, seed=91, fault="camera_noise")
    generate(second, steps=30, seed=91, fault="camera_noise")
    assert (first / "measurements.csv").read_bytes() == (second / "measurements.csv").read_bytes()
    assert (first / "truth.csv").read_bytes() == (second / "truth.csv").read_bytes()


def test_evaluator_reports_position_rmse_and_availability(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks.csv"
    truth = tmp_path / "truth.csv"
    tracks.write_text("step,time_s,target_id,x,y,vx,vy\n0,0,0,3,4,0,0\n", encoding="utf-8")
    truth.write_text("step,time_s,target_id,x,y\n0,0,0,0,0\n1,0.1,0,0,0\n", encoding="utf-8")
    metrics = evaluate(tracks, truth)
    assert metrics["position_rmse"] == pytest.approx(5.0)
    assert metrics["track_availability"] == pytest.approx(0.5)


def test_monitor_inflates_covariance_for_degraded_measurements(tmp_path: Path) -> None:
    measurements = tmp_path / "measurements.csv"
    innovations = tmp_path / "innovations.csv"
    with measurements.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "target_id", "sensor_id", "x", "y", "sigma"])
        for step in range(8):
            writer.writerow([step, 0, "gps", 0, 0, 0.5])
            writer.writerow([step, 0, "camera", 0, 0, 0.9])
    with innovations.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "target_id", "sensor_id", "dx", "dy", "sxx", "syy"])
        for step in range(8):
            writer.writerow([step, 0, "gps", 0, 0, 1, 1])
            writer.writerow([step, 0, "camera", 8, 8, 1, 1])
    adjusted, health = tmp_path / "adjusted.csv", tmp_path / "health.csv"
    monitor(measurements, innovations, adjusted, health, window_steps=3)
    with adjusted.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    last_camera = [row for row in rows if row["sensor_id"] == "camera"][-1]
    assert float(last_camera["sigma"]) > 0.9
    with health.open(newline="", encoding="utf-8") as stream:
        last_health = [row for row in csv.DictReader(stream) if row["sensor_id"] == "camera"][-1]
    assert "elevated_innovation_variance" in last_health["reason"]


def test_knn_rul_inference_returns_nonnegative_estimate() -> None:
    features = ["mean_nis", "bias_norm", "delivery_rate", "nis_trend_10",
                "bias_trend_10", "sensor_is_camera", "sensor_is_radar"]
    model = {
        "model_type": "knn_regressor", "features": features,
        "mean": [0.0] * 7, "scale": [1.0] * 7,
        "support_vectors": [[0.0] * 7, [1.0] * 7],
        "remaining_seconds": [10.0, 0.0], "neighbors": 1,
    }
    prediction, spread = predict_one(model, [0.0] * 7)
    assert prediction == pytest.approx(10.0)
    assert spread == pytest.approx(0.0)


def test_cpp_tracker_cli_produces_estimates(tmp_path: Path) -> None:
    tracker = Path(os.environ.get("TRACKER_EXECUTABLE", ""))
    if not tracker.is_absolute():
        tracker = ROOT / tracker
    if not tracker.is_file():
        pytest.skip("compiled C++ tracker not available")
    measurements = tmp_path / "measurements.csv"
    measurements.write_text(
        "step,target_id,sensor_id,x,y,sigma\n"
        "0,0,gps,1,2,0.5\n1,0,gps,1.1,2.1,0.5\n",
        encoding="utf-8",
    )
    tracks, innovations = tmp_path / "tracks.csv", tmp_path / "innovations.csv"
    subprocess.run([str(tracker), str(measurements), str(tracks), str(innovations), "0.1"], check=True)
    with tracks.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[-1]["target_id"] == "0"
    assert innovations.read_text(encoding="utf-8").startswith("step,target_id,sensor_id")


def test_sweep_summary_handles_single_and_empty_samples() -> None:
    result = summarize([2.0, 4.0])
    assert result["count"] == 2
    assert result["mean"] == 3.0
    assert result["stddev"] == pytest.approx(2**0.5)
    assert result["median"] == 3.0
    assert result["ci95_low"] <= result["mean"] <= result["ci95_high"]
    assert summarize([2.0, 4.0]) == result
    assert summarize([]) == {
        "count": 0, "mean": None, "stddev": None, "median": None,
        "ci95_low": None, "ci95_high": None,
    }


def test_latest_sweep_returns_newest_saved_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    older = data / "sweeps" / "older" / "sweep_summary.json"
    newer = data / "sweeps" / "nested" / "newer" / "sweep_summary.json"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text('{"seed_start": 1}', encoding="utf-8")
    newer.write_text('{"seed_start": 50}', encoding="utf-8")
    os.utime(newer, (older.stat().st_mtime + 2, older.stat().st_mtime + 2))
    monkeypatch.setattr(api, "ROOT", tmp_path)

    result = api.latest_sweep()

    assert result["available"] is True
    assert result["summary"] == {"seed_start": 50}
    assert result["output_directory"] == str(Path("data") / "sweeps" / "nested" / "newer")


def test_latest_sweep_reports_missing_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "ROOT", tmp_path)
    assert api.latest_sweep() == {"available": False, "summary": None}
