"""Export showcase checkpoints for a fast dashboard; never train inside the app."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay, cKDTree
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Interpolation": "interpolation",
    "Random Forest": "random_forest",
    "MLP": "mlp",
    "Polynomial Ridge": "polynomial_ridge",
}


def notebook_definitions():
    """Reuse the notebook's exact inference/physics definitions without training cells."""
    notebook = json.loads(
        (ROOT / "notebooks/flood_reconstruction_model_comparison.ipynb").read_text()
    )
    scope = dict(np=np, torch=torch, nn=nn, interp1d=interp1d, Delaunay=Delaunay, cKDTree=cKDTree)
    wanted = {
        "clean_wave",
        "CoordinateMLP",
        "interpolation_field",
        "predict_mlp",
        "initial_height",
        "rusanov_flux",
        "solve_reference",
        "xt_coordinates",
        "ShallowWaterNN",
        "physics_residual",
    }
    found = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(
            line
            for line in "".join(cell["source"]).splitlines()
            if not line.lstrip().startswith("%")
        )
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]), "showcase-definitions", "exec"
                    ),
                    scope,
                )
                found.add(node.name)
    if found != wanted:
        raise ValueError(f"Missing notebook definitions: {wanted - found}")
    return scope


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_scores(reference, predictions, csv_path):
    scores = pd.read_csv(csv_path).set_index("Model")
    for name, key in MODELS.items():
        rmse = float(np.sqrt(np.mean((predictions[key] - reference) ** 2)) * 1000)
        np.testing.assert_allclose(
            rmse,
            scores.loc[name, "RMSE (mm)"],
            rtol=1e-6,
            atol=1e-5,
            err_msg=f"Dashboard does not match showcase: {name}",
        )


def export(run, destination):
    torch.set_num_threads(2)
    scope = notebook_definitions()
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(run / "split.npz", allow_pickle=False) as split:
        for key in split.files:
            scope[key] = split[key].copy()
    waves = {}
    for number in (1, 3):
        with np.load(ROOT / f"reports/artifacts/O4_wave{number}.npz", allow_pickle=False) as data:
            waves[number] = scope["clean_wave"]({key: data[key] for key in data.files})
    wave = waves[1]
    scope.update(xy=wave["xy"], times=wave["time_s"], water=wave["wse_m"])
    checkpoint = torch.load(run / "mlp.pt", map_location="cpu", weights_only=True)
    scope["mlp"] = scope["CoordinateMLP"]()
    scope["mlp"].load_state_dict(checkpoint["state_dict"])
    scope["feature_scaler"] = MinMaxScaler(feature_range=(-1, 1)).fit(
        np.array([checkpoint["feature_min"], checkpoint["feature_max"]])
    )
    scope.update(target_mean=checkpoint["target_mean"], target_std=checkpoint["target_std"])
    regressors = {
        name: joblib.load(run / f"{key}.joblib")
        for name, key in MODELS.items()
        if name in ("Random Forest", "Polynomial Ridge")
    }
    for number, wave in waves.items():
        if number == 1:
            ids = scope["heldout_ids"]
            xy, times = wave["xy"][ids], wave["time_s"][scope["test_frames"]]
            reference = wave["wse_m"][np.ix_(scope["test_frames"], ids)]
            sensor_xy = wave["xy"][scope["sensor_ids"]]
        else:
            distance, _ = cKDTree(scope["xy"]).query(wave["xy"])
            common = distance <= 1e-7
            time_mask = (wave["time_s"] >= scope["times"][scope["train_frames"]].min()) & (
                wave["time_s"] <= scope["times"][scope["train_frames"]].max()
            )
            xy, times = wave["xy"][common], wave["time_s"][time_mask]
            reference = wave["wse_m"][time_mask][:, common]
            sensor_xy = np.empty((0, 2))  # No wave 3 fitting observations.
        features = np.column_stack([np.tile(xy, (len(times), 1)), np.repeat(times, len(xy))])
        predictions = {
            MODELS[name]: model.predict(features).reshape(reference.shape)
            for name, model in regressors.items()
        }
        predictions["mlp"] = scope["predict_mlp"](features).reshape(reference.shape)
        predictions["interpolation"] = scope["interpolation_field"](xy, times)
        csv_name = "test_results.csv" if number == 1 else "wave3_transfer_results.csv"
        check_scores(reference, predictions, ROOT / "reports/notebook_results" / csv_name)
        if number == 1:
            with np.load(run / "test_predictions.npz", allow_pickle=False) as saved:
                for key in predictions:
                    np.testing.assert_allclose(
                        predictions[key].ravel(), saved[key], rtol=1e-5, atol=1e-7
                    )
        np.savez_compressed(
            destination / f"wave{number}.npz",
            xy_m=xy,
            time_s=times,
            reference_m=reference,
            sensor_xy_m=sensor_xy,
            **predictions,
        )

    reference = scope["solve_reference"]()
    scope["st"] = reference["time"]
    sx, times = reference["x"], reference["time"]
    frames = np.arange(len(times))
    test_frames = frames[(frames > 0) & (frames < len(times) - 1) & (frames % 5 == 4)]
    sensor_ids = np.linspace(6, len(sx) - 7, 12, dtype=int)
    heldout = np.setdiff1d(np.arange(len(sx)), sensor_ids)
    query = scope["xt_coordinates"](sx, times[test_frames])
    diagnostic = torch.rand((1024, 2), generator=torch.Generator().manual_seed(45))
    diagnostic[:, 0] = sx[0] + diagnostic[:, 0] * (sx[-1] - sx[0])
    diagnostic[:, 1] *= times[-1]
    out = {
        "x": sx,
        "time": times[test_frames],
        "reference": reference["state"][test_frames],
        "heldout_ids": heldout,
        "diagnostic_xt": diagnostic.numpy(),
    }
    scores = pd.read_csv(ROOT / "reports/notebook_results/synthetic_results.csv").set_index("Model")
    for name, key in (("Data-only NN", "data_only"), ("PINN", "pinn")):
        model = scope["ShallowWaterNN"]()
        model.load_state_dict(
            torch.load(
                run / ("data-only_nn.pt" if key == "data_only" else "pinn.pt"),
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        with torch.no_grad():
            out[key] = model(query).numpy().reshape(len(test_frames), len(sx), 2)
        residual = scope["physics_residual"](model, diagnostic.clone()).detach().numpy()
        out[key + "_residual"] = residual
        for i, column in enumerate(("Height RMSE", "Discharge RMSE")):
            rmse = np.sqrt(
                np.mean((out[key][:, heldout, i] - out["reference"][:, heldout, i]) ** 2)
            )
            np.testing.assert_allclose(rmse, scores.loc[name, column], rtol=1e-5)
        for i, column in enumerate(("Continuity RMS", "Momentum RMS")):
            np.testing.assert_allclose(
                np.sqrt(np.mean(residual[:, i] ** 2)), scores.loc[name, column], rtol=1e-5
            )
    np.savez_compressed(destination / "synthetic.npz", **out)
    inputs = [
        run / name
        for name in (
            "mlp.pt",
            "random_forest.joblib",
            "polynomial_ridge.joblib",
            "pinn.pt",
            "data-only_nn.pt",
            "split.npz",
            "run_config.json",
        )
    ]
    manifest = {
        "schema_version": 1,
        "run_config": json.loads((run / "run_config.json").read_text()),
        "inputs": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "files": {
            p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(destination.glob("*.npz"))
        },
        "measured_source": "https://zenodo.org/records/17777387",
        "measured_license": "CC BY 4.0",
        "notebook_sha256": digest(ROOT / "notebooks/flood_reconstruction_model_comparison.ipynb"),
        "score_verification": "All measured and synthetic metrics match reports/notebook_results",
        "synthetic_reference_diagnostics": reference["diagnostics"],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported and verified showcase predictions: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/kaggle_notebook")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/dashboard")
    args = parser.parse_args()
    export(args.run.resolve(), args.output.resolve())
