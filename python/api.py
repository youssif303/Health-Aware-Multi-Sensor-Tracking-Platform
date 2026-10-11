"""FastAPI service for launching and retrieving tracking experiments."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
TRACKER = Path(os.environ.get("TRACKER_EXECUTABLE", ROOT / "build" / "tracker.exe")).resolve()
RUL_MODEL = Path(os.environ.get("RUL_MODEL_PATH", ROOT / "models" / "rul_knn.json")).resolve()
EXPERIMENT_SCRIPT = ROOT / "python" / "run_experiment.py"
RUNS_ROOT = ROOT / "data" / "api"

app = FastAPI(title="Health-Aware Multi-Sensor Tracking API", version="0.1.0")


class ExperimentRequest(BaseModel):
    fault: Literal["none", "camera_noise", "radar_bias", "gps_dropout", "camera_ramp", "radar_ramp"] = "camera_noise"
    seed: int = 7
    steps: int = Field(default=500, ge=120, le=5000)
    onset_step: int | None = Field(default=None, ge=0)
    failure_step: int | None = Field(default=None, ge=1)


def latest_rows(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        result: dict[str, dict[str, str]] = {}
        for row in csv.DictReader(stream):
            result[row[key]] = row
        return result


def experiment_directory(run_id: str) -> Path:
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise HTTPException(status_code=404, detail="experiment not found")
    output = RUNS_ROOT / run_id
    if not (output / "metrics.json").is_file():
        raise HTTPException(status_code=404, detail="experiment not found")
    return output


@app.get("/health")
def service_health() -> dict[str, object]:
    return {
        "status": "ok",
        "tracker_available": TRACKER.is_file(),
        "rul_model_available": RUL_MODEL.is_file(),
    }


@app.post("/experiments", status_code=201)
def create_experiment(request: ExperimentRequest) -> dict[str, object]:
    if not TRACKER.is_file():
        raise HTTPException(status_code=503, detail="C++ tracker is not built or TRACKER_EXECUTABLE is invalid")
    onset = request.onset_step if request.onset_step is not None else request.steps // 2
    failure = request.failure_step if request.failure_step is not None else request.steps - 1
    if onset >= request.steps or failure >= request.steps or failure <= onset:
        raise HTTPException(status_code=422, detail="onset_step and failure_step must be ordered within the run")
    run_id = uuid.uuid4().hex
    output = RUNS_ROOT / run_id
    command = [
        sys.executable, str(EXPERIMENT_SCRIPT), "--tracker", str(TRACKER),
        "--output", str(output), "--steps", str(request.steps),
        "--seed", str(request.seed), "--fault", request.fault,
    ]
    if request.onset_step is not None:
        command.extend(["--onset-step", str(request.onset_step)])
    if request.failure_step is not None:
        command.extend(["--failure-step", str(request.failure_step)])
    if request.fault in {"camera_ramp", "radar_ramp"} and RUL_MODEL.is_file():
        command.extend(["--rul-model", str(RUL_MODEL)])
    try:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   check=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="experiment exceeded the 120 second limit") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "experiment failed").strip()[-2000:]
        raise HTTPException(status_code=500, detail=detail) from exc
    metrics_path = output / "metrics.json"
    if not metrics_path.is_file():
        raise HTTPException(status_code=500, detail="experiment completed without metrics.json")
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {"run_id": run_id, "metrics": result, "output_directory": str(output.relative_to(ROOT)),
            "log": completed.stdout[-1000:]}


@app.get("/experiments")
def list_experiments(limit: int = Query(default=10, ge=1, le=50)) -> dict[str, object]:
    if not RUNS_ROOT.is_dir():
        return {"runs": []}
    candidates = sorted((path for path in RUNS_ROOT.iterdir() if path.is_dir()),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    runs = []
    for path in candidates:
        metrics_path = path / "metrics.json"
        if path.name and len(path.name) == 32 and metrics_path.is_file():
            runs.append({"run_id": path.name, "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))})
            if len(runs) >= limit:
                break
    return {"runs": runs}


@app.get("/sweeps/latest")
def latest_sweep() -> dict[str, object]:
    candidates = []
    for data_root in (ROOT / "data", ROOT / "results"):
        if data_root.is_dir():
            candidates.extend(data_root.rglob("sweep_summary.json"))
    if not candidates:
        return {"available": False, "summary": None}
    path = max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return {"available": True, "summary": json.loads(path.read_text(encoding="utf-8")),
            "output_directory": str(path.parent.relative_to(ROOT))}


@app.get("/experiments/{run_id}")
def get_experiment(run_id: str) -> dict[str, object]:
    output = experiment_directory(run_id)
    metrics_path = output / "metrics.json"
    with (output / "health.csv").open(newline="", encoding="utf-8") as stream:
        health_rows = list(csv.DictReader(stream))
    events: list[dict[str, object]] = []
    active: dict[tuple[str, str], dict[str, object]] = {}
    for row in health_rows:
        reason = row["reason"]
        if reason == "healthy":
            continue
        key = (row["sensor_id"], reason)
        step = int(row["step"])
        event = active.get(key)
        if event and step == int(event["last_step"]) + 1:
            event["last_step"] = step
        else:
            event = {"sensor_id": row["sensor_id"], "reason": reason,
                     "first_step": step, "last_step": step}
            events.append(event)
            active[key] = event
    return {
        "run_id": run_id,
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "latest_sensor_health": latest_rows(output / "health.csv", "sensor_id"),
        "latest_rul_predictions": latest_rows(output / "rul_predictions.csv", "sensor_id"),
        "fault_events": events,
        "output_directory": str(output.relative_to(ROOT)),
    }


@app.get("/experiments/{run_id}/tracks")
def get_tracks(run_id: str, limit: int = Query(default=120, ge=20, le=1000)) -> dict[str, object]:
    output = experiment_directory(run_id)
    estimates: dict[str, list[dict[str, float | int]]] = {}
    path = output / "adaptive_tracks.csv"
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                target = row["target_id"]
                estimates.setdefault(target, []).append({
                    "step": int(row["step"]), "x": float(row["x"]), "y": float(row["y"]),
                })
    truth: dict[str, list[dict[str, float | int]]] = {}
    truth_path = output / "truth.csv"
    if truth_path.is_file():
        with truth_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                target = row["target_id"]
                truth.setdefault(target, []).append({
                    "step": int(row["step"]), "x": float(row["x"]), "y": float(row["y"]),
                })
    return {"estimates": {key: values[-limit:] for key, values in estimates.items()},
            "truth": {key: values[-limit:] for key, values in truth.items()}}


app.mount("/", StaticFiles(directory=ROOT / "dashboard", html=True), name="dashboard")
