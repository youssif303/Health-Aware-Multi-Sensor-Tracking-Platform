# Health-Aware Multi-Sensor Tracking Platform

A simulation and tracking platform that fuses noisy measurements from multiple sensors, detects sensor degradation, and adjusts the tracker so faulty sensors have less influence.

## First milestone

Build a reproducible 2D simulation with three moving drones and three sensor types. Keep the first version intentionally measurable and small:

- **Tracking core (C++):** constant-velocity Kalman filter per drone, with a shared state convention `[x, y, vx, vy]`.
- **Sensors:** radar-like position fixes, camera-like position fixes, and GPS-like position fixes. Each has configurable noise, bias, and packet dropouts. These are simplified observation models for the MVP; realistic range/bearing radar and image detections can follow once the end-to-end loop works.
- **Health monitoring (Python):** derive residual features from sensor measurements versus track predictions; detect rising noise, persistent bias, and missing data. Start with transparent statistical detectors and establish labeled simulated faults before training predictive models.
- **Adaptive fusion:** turn health estimates into per-sensor measurement covariance inflation (and skip updates during outages), rather than relying on a hard-coded sensor ranking.
- **Evaluation:** compare the same seeded scenarios with health adaptation disabled and enabled. Report position RMSE, track availability, fault detection delay, and false alarm rate. State the scenario and seeds alongside any percentage improvement.
- **Service and UI:** expose simulation and evaluation through FastAPI; show tracks, sensor health, fault events, and metrics in a small dashboard.
- **Reproducibility:** log run parameters, metrics, and model versions with MLflow; provide a Docker Compose setup and CI checks after the MVP runs locally.

## Suggested repository layout

```text
cpp/                 Kalman filter library and native bindings/CLI
python/              Simulator, health monitor, RUL model, evaluator, API
dashboard/           Health and tracking visualization
configs/             Versioned scenario and fault profiles
experiments/         Evaluation entry points and MLflow setup
models/              Trained health/RUL model artifacts
tests/               C++ and Python tests
docker/              Container and Compose configuration
```

The implementation lives in `cpp/` and `python/`. It deliberately keeps
measurements and ground truth in separate CSV files. The tracker only reads
measurements and nominal sensor noise; it does not receive fault labels or
truth. Sensor updates are processed from lower to higher nominal noise, so the
health monitor can compare later updates against the higher-confidence sensors
from the same time step. Coordinates are in meters, the simulation interval is
0.1 seconds, and the first state convention is `[x, y, vx, vy]` in meters and
meters per second.

## Run the current MVP

Generate a clean scenario:

```powershell
python python/simulate.py --output data/clean --seed 7 --fault none
```

Generate a scenario with a camera noise fault after halfway:

```powershell
python python/simulate.py --output data/camera_fault --seed 7 --fault camera_noise
```

Build the C++ tracker (CMake and a C++17 compiler required):

```powershell
cmake -S cpp -B build
cmake --build build --config Release
```

For the default Ninja generator, the executable is `build\tracker.exe`. Run:

```powershell
.\build\tracker.exe data\camera_fault\measurements.csv data\camera_fault\tracks.csv 0.1
python python/evaluate.py data/camera_fault/tracks.csv data/camera_fault/truth.csv
```

For a Visual Studio generator, use `build\Release\tracker.exe`. The tracker
emits one state estimate per target and time step in `tracks.csv`.

Run a complete matched baseline/adaptive experiment (Ninja example):

```powershell
python python/run_experiment.py --tracker build\tracker.exe --output data\camera_fault --seed 7 --fault camera_noise
```

The experiment saves both track outputs, pre-update innovations, per-sensor
health history, an adapted measurement file, and `metrics.json`. The health
monitor uses a rolling window of normalized innovation variance, persistent
innovation bias, and measurement delivery rate. It inflates covariance when
arriving measurements look unreliable; missing packets are skipped and reported
as reduced delivery health without penalizing the packets that do arrive. This
is a transparent statistical baseline used for online trust adjustment.

Initial matched replay results for seed 7 and 500 steps (one simulated fault
starts halfway through the run):

| Fault | Baseline position RMSE | Adaptive position RMSE | RMSE improvement | Detection delay | False alarm rate |
|---|---:|---:|---:|---:|---:|
| Camera noise x5 | 0.464 | 0.305 | 34.1% | 0.1 s | 0.16% |
| Radar +12 m bias | 0.560 | 0.304 | 45.7% | 0.1 s | 0.56% |
| GPS 75% packet loss | 0.329 | 0.330 | -0.4% | 0.1 s | 0.00% |

These are early simulator results from a single seed, not a general performance
claim. GPS packet loss is detected, but the adaptive filter does not improve
error because it already skips missing packets and the packets that arrive are
still accurate. Repeat across many seeds and fault severities before making a
headline claim.

Run a matched multi-seed sweep to see average gains, run-to-run variation,
and deterministic percentile-bootstrap confidence intervals:

```powershell
python python/sweep_experiments.py --tracker build\tracker.exe --output data\sweep --runs-per-fault 10 --seed-start 7 --steps 500
```

The sweep writes one row per run to `sweep_runs.csv` and per-fault mean,
standard deviation, median, and 95% confidence interval metrics to
`sweep_summary.json`. Confidence intervals use 10,000 bootstrap resamples of
the per-seed paired RMSE improvement. The default
fault set includes gradual camera and radar degradation, so the sweep also
evaluates remaining-life predictions. Select a smaller or different set with
`--faults camera_ramp radar_ramp`. When `MLFLOW_TRACKING_URI` is configured,
the sweep aggregate and each individual experiment are logged to MLflow.

A quick three-seed smoke sweep (seeds 40–42, 500 steps) produced these mean
matched RMSE changes:

| Fault | Baseline RMSE | Adaptive RMSE | Mean RMSE improvement | Run-to-run SD |
|---|---:|---:|---:|---:|
| Camera noise x5 | 0.474 m | 0.311 m | 34.5% | 1.7 percentage points |
| Radar +12 m bias | 0.547 m | 0.307 m | 43.9% | 0.1 percentage points |
| GPS 75% packet loss | 0.341 m | 0.342 m | -0.5% | 0.1 percentage points |

This is a small sanity sweep, not enough trials to claim broad performance.
It confirms the expected behavior: down-weighting bad camera/radar data helps;
down-weighting GPS packets that do arrive accurately does not help tracking.
The same three-seed sweep also measured gradual faults: camera-ramp tracking
RMSE improved 19.6% with 3.27 s mean RUL MAE, and radar-ramp RMSE improved
19.3% with 2.75 s mean RUL MAE. The dashboard shows the summary from the most
recent sweep, including detection delay and RUL error.

For a stronger evaluation in the running Docker stack, run 30 seeds for each
of the five fault profiles (150 matched experiments total):

```powershell
docker compose exec -T api python /app/python/sweep_experiments.py --tracker /usr/local/bin/tracker --output /app/data/sweep_30seed --runs-per-fault 30 --seed-start 100 --steps 500
```

The dashboard automatically loads the most recently written sweep summary.

The checked-in benchmark in `results/sweep_30seed/` contains 30 matched seeds
for each fault (seeds 100–129, 500 steps). Adaptive fusion improved four of the
five profiles. GPS dropout remained a known limitation: missing packets are
already skipped, and the arriving GPS measurements are still accurate.

| Fault profile | Baseline RMSE | Adaptive RMSE | Paired mean improvement (95% CI) |
|---|---:|---:|---:|
| Camera noise | 0.474 m | 0.304 m | 35.67% (34.64%, 36.66%) |
| Radar bias | 0.551 m | 0.304 m | 44.85% (44.38%, 45.30%) |
| GPS dropout | 0.340 m | 0.342 m | -0.57% (-0.65%, -0.49%) |
| Camera ramp | 0.365 m | 0.296 m | 18.88% (17.85%, 19.85%) |
| Radar ramp | 0.361 m | 0.288 m | 20.06% (19.55%, 20.57%) |

The confidence intervals are percentile-bootstrap intervals over paired
per-seed RMSE improvements (10,000 resamples). These are results from the
synthetic simulator and do not establish performance on real sensor hardware.

## Train the first remaining-life model

Generate gradual camera-noise and radar-bias histories, with the simulated
failure threshold defined as camera noise reaching 5x nominal or radar bias
reaching 10 m. The labels are time remaining until that threshold:

```powershell
python python/generate_rul_dataset.py --tracker build\tracker.exe --output data\rul_training --scenarios 40 --steps 500 --seed 100
python python/train_rul_model.py data\rul_training\rul_dataset.csv --model models\rul_knn.json --seed 23
python python/predict_rul.py models\rul_knn.json --sensor camera --mean-nis 3 --bias-norm 0.5 --delivery-rate 1 --nis-trend-10 0.5 --bias-trend-10 0.1
```

The first model is a k-nearest-neighbors regressor trained on normalized
innovation features. Its held-out split is by scenario, so samples from one
simulated flight cannot appear in both training and evaluation. With 40
generated scenarios (seeds 100–139), 30 training scenarios, and 10 held-out
scenarios, the model reached 3.29 s MAE versus 7.30 s for a median-RUL baseline
(55.0% lower MAE). See [rul_model_metrics.json](models/rul_model_metrics.json)
and the trained artifact at [rul_knn.json](models/rul_knn.json).
If `MLFLOW_TRACKING_URI` is set, the training command also logs its parameters,
held-out metrics, dataset, model, and metrics JSON to the configured MLflow
experiment.

This result is limited to the synthetic ramp profiles and thresholds above. It
does not establish performance on real sensor hardware or degradation patterns
outside the training data.

Run a gradual-degradation experiment with per-step RUL predictions and compare
those predictions with the known simulated threshold time:

```powershell
python python/run_experiment.py --tracker build\tracker.exe --output data\rul_demo --seed 31 --steps 500 --fault camera_ramp --onset-step 120 --failure-step 390 --rul-model models\rul_knn.json
```

## API

Install the Python dependencies and start the service after building the
tracker:

```powershell
python -m pip install -r requirements.txt
cmake -S cpp -B build
cmake --build build --config Release
python -m uvicorn api:app --app-dir python --reload
```

The service hosts the dashboard at `/` and provides `GET /health`,
`GET /experiments`, `POST /experiments`, `GET /experiments/{run_id}`, and
`GET /experiments/{run_id}/tracks`. Experiment requests are limited to a supported
fault type, seed, and bounded number of steps; each run receives a generated ID
and writes its results under `data/api/`. Configure the tracker executable
with `TRACKER_EXECUTABLE` and the optional RUL model with `RUL_MODEL_PATH`.

## Docker Compose

Build and start the complete local stack:

```powershell
docker compose up --build
```

Open the dashboard at `http://localhost:8000` and the MLflow UI at
`http://localhost:5001`. The API waits until MLflow is healthy; run metrics and
health/RUL files are logged to the `health-aware-tracking` experiment. Docker
stores run outputs and MLflow metadata in named volumes. Both services bind to
localhost by default.

## Render deployment

`render.yaml` defines one free web service in Frankfurt. It runs the public
dashboard/API and a loopback-only MLflow tracking server in the same container,
because Render's Free web services cannot receive private network traffic and
private services are paid. The API dashboard includes the checked-in benchmark
summary. To deploy, connect the GitHub repository in Render and create a
Blueprint from `render.yaml`.

Free services sleep after 15 minutes without traffic and wake on the next
request (typically taking about a minute). Their filesystem is ephemeral, so
new experiment outputs and MLflow runs/artifacts are lost when the service
restarts, sleeps, or redeploys; only checked-in results persist. Render provides
750 free instance hours per workspace each month, subject to its included
bandwidth and build limits. This free deployment is suitable for a demo, not
durable production data.

## CI checks

GitHub Actions builds the C++ tracker, runs the Python suite, validates the
Compose file, and builds the application image on pushes and pull requests.
Run the C++ and Python checks locally after installing development
dependencies:

```powershell
python -m pip install -r requirements-dev.txt
cmake -S cpp -B build
cmake --build build --config Release
$env:PYTHONPATH = "python"
$env:TRACKER_EXECUTABLE = "build\tracker.exe"
python -m pytest -q
```

## Build sequence

1. Define simulation data formats, coordinate units, time step, and a fixed-seed baseline scenario.
2. Implement and validate the C++ Kalman filter against generated trajectories.
3. Add sensor noise, bias drift, noise growth, and outages; save measurements and ground truth in a documented format.
4. Establish a no-health-monitor baseline and metrics.
5. Add residual-based health features and fault detection, then feed health into measurement covariance.
6. Compare matched runs across many seeds and fault severities; the first synthetic RUL model is now available as a baseline.
7. Add a dashboard, MLflow logging, Docker packaging, and CI around the API and experiment workflow.

## Key design choices

- Keep ground truth separate from sensor measurements; evaluation code may use truth, while the online health monitor may not.
- Use identical random seeds and trajectories for baseline/adaptive comparisons so the measured delta is attributable to health adaptation.
- Treat remaining-useful-life as a supervised prediction problem with an explicit failure threshold. The initial model is synthetic-only and needs validation on broader scenarios before use beyond this simulator.
- Make sensor health observable and auditable: expose the residual history, health score, and reason for each trust adjustment.

## Initial success criteria

- The tracker fuses asynchronous, noisy measurements for three targets without using ground truth at runtime.
- Injected bias, increasing noise, and dropouts produce visible health changes and bounded recovery behavior.
- On a documented set of fault scenarios, adaptive fusion reduces position RMSE versus the matched no-monitor baseline without an unacceptable rise in false alarms or track loss.
- A single documented command can run an experiment and emit machine-readable metrics.
