"""Array contracts and metrics shared by the dashboard and its scientific checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/dashboard"
MODELS = {
    "Interpolation": "interpolation",
    "Random Forest": "random_forest",
    "MLP": "mlp",
    "Polynomial Ridge": "polynomial_ridge",
}


def load_archive(name):
    manifest = json.loads((DATA / "manifest.json").read_text())
    path = DATA / name
    if name not in manifest["files"]:
        raise ValueError(f"Unregistered dashboard artifact: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
        raise ValueError(f"Dashboard artifact checksum mismatch: {name}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_wave(number):
    if number not in (1, 3):
        raise ValueError("Only the measured O4 waves 1 and 3 are available")
    wave = load_archive(f"wave{number}.npz")
    expected = (len(wave["time_s"]), len(wave["xy_m"]))
    for key in ("reference_m", *MODELS.values()):
        if wave[key].shape != expected or not np.isfinite(wave[key]).all():
            raise ValueError(f"Invalid measured field: {key}")
    return wave


def scores(reference, prediction):
    """Measured WSE comes in meters; all displayed errors use millimeters."""
    error = (np.asarray(prediction) - np.asarray(reference)) * 1000
    return {
        "RMSE (mm)": float(np.sqrt(np.mean(error**2))),
        "MAE (mm)": float(np.mean(np.abs(error))),
        "95th abs. error (mm)": float(np.percentile(np.abs(error), 95)),
    }


def frame_table(wave, model, frame):
    if model not in MODELS or not 0 <= frame < len(wave["time_s"]):
        raise ValueError("Unknown model or observation frame")
    ref, pred = wave["reference_m"][frame], wave[MODELS[model]][frame]
    return pd.DataFrame(
        {
            "x_m": wave["xy_m"][:, 0],
            "y_m": wave["xy_m"][:, 1],
            "time_s": wave["time_s"][frame],
            "measured_wse_mm": ref * 1000,
            "predicted_wse_mm": pred * 1000,
            "absolute_error_mm": np.abs(pred - ref) * 1000,
        }
    )


def measured_results(number):
    file = "test_results.csv" if number == 1 else "wave3_transfer_results.csv"
    return pd.read_csv(ROOT / "reports/notebook_results" / file)


def sensor_results():
    return pd.read_csv(ROOT / "reports/sensor_ablation/runs.csv")


def summarize_sensors(runs):
    """Sample SD over seeds, not a confidence interval across independent floods."""
    return (
        runs.groupby(["Sensors", "Model"])["RMSE (mm)"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def elevation_limits(wave):
    values = [wave[key] for key in ("reference_m", *MODELS.values())]
    return min(float(v.min()) for v in values) * 1000, max(float(v.max()) for v in values) * 1000
