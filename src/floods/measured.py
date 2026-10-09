"""Sparse measured WSE reconstruction; no unobserved velocities or depths."""

import copy
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay, cKDTree
import torch

from .models import MLP


def split_indices(frames: int, points: int, sensors: int, seed: int) -> dict[str, np.ndarray]:
    if not 3 <= sensors < points:
        raise ValueError("Sensor count must be at least 3 and smaller than valid spatial point count")
    if frames < 10:
        raise ValueError("At least 10 frames required for temporal holdouts")
    ids = np.arange(frames)
    interior = (ids > 0) & (ids < frames - 1)
    val = ids[interior & (ids % 5 == 2)]
    test = ids[interior & (ids % 5 == 4)]
    train = np.setdiff1d(ids, np.concatenate([val, test]))
    chosen = np.sort(np.random.default_rng(seed).choice(points, sensors, replace=False))
    return {"train_times": train, "validation_times": val, "test_times": test,
            "sensor_points": chosen, "heldout_points": np.setdiff1d(np.arange(points), chosen)}


def make_coordinates(xy: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.column_stack([np.tile(xy, (len(times), 1)), np.repeat(times, len(xy))])


def errors(predicted: np.ndarray, observed: np.ndarray) -> dict:
    residual_mm = (predicted - observed) * 1000
    return {"rmse_mm": float(np.sqrt(np.mean(residual_mm ** 2))),
            "mae_mm": float(np.mean(np.abs(residual_mm))),
            "p95_absolute_error_mm": float(np.percentile(np.abs(residual_mm), 95))}


def interpolate_space(xy: np.ndarray, sensor_ids: np.ndarray, sensor_values: np.ndarray) -> tuple[np.ndarray, int]:
    """Linear interpolation on a triangulation; nearest fallback outside its hull."""
    triangulation = Delaunay(xy[sensor_ids])
    simplex = triangulation.find_simplex(xy)
    inside = simplex >= 0
    result = np.empty((len(sensor_values), len(xy)))
    transform = triangulation.transform[simplex[inside]]
    first = np.einsum("nij,nj->ni", transform[:, :2, :], xy[inside] - transform[:, 2, :])
    weights = np.column_stack([first, 1 - first.sum(axis=1)])
    vertices = triangulation.simplices[simplex[inside]]
    result[:, inside] = np.einsum("tnv,nv->tn", sensor_values[:, vertices], weights)
    if np.any(~inside):
        closest = cKDTree(xy[sensor_ids]).query(xy[~inside])[1]
        result[:, ~inside] = sensor_values[:, closest]
    return result, int((~inside).sum())


def run_measured(dataset: Path, output: Path, epochs: int = 1500, sensors: int = 128, seed: int = 42) -> dict:
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(dataset, allow_pickle=False) as data:
        xy, times, water = data["xy"], data["time_s"], data["wse_m"]
        metadata = json.loads(str(data["metadata"]))
    split = split_indices(len(times), len(xy), sensors, seed)
    (output / "split.json").write_text(json.dumps({k: v.tolist() for k, v in split.items()}, indent=2) + "\n")
    training_times = split["train_times"]
    sensor_ids = split["sensor_points"]
    train_coordinates = make_coordinates(xy[sensor_ids], times[training_times])
    train_targets = water[np.ix_(training_times, sensor_ids)].reshape(-1, 1)
    lower = train_coordinates.min(axis=0)
    scale = np.maximum(train_coordinates.max(axis=0) - lower, 1e-8)
    mean, std = float(train_targets.mean()), max(float(train_targets.std()), 1e-6)
    inputs = torch.tensor(2 * (train_coordinates - lower) / scale - 1, dtype=torch.float32)
    targets = torch.tensor((train_targets - mean) / std, dtype=torch.float32)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = MLP(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def predict(xy_query: np.ndarray, t_query: np.ndarray) -> np.ndarray:
        coordinates = make_coordinates(xy_query, t_query)
        result = []
        with torch.no_grad():
            for chunk in np.array_split(coordinates, max(1, int(np.ceil(len(coordinates) / 8192)))):
                normalized = torch.tensor(2 * (chunk - lower) / scale - 1, dtype=torch.float32)
                result.append(model(normalized).numpy()[:, 0] * std + mean)
        return np.concatenate(result).reshape(len(t_query), len(xy_query))

    heldout = split["heldout_points"]
    # A fixed subset of validation locations limits laptop runtime; test uses all holdout locations.
    validation_points = heldout[::max(1, len(heldout) // 400)]
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history, start = [], time.monotonic()
    for epoch in range(1, epochs + 1):
        ids = torch.randint(len(inputs), (min(512, len(inputs)),))
        optimizer.zero_grad()
        loss = torch.mean((model(inputs[ids]) - targets[ids]) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch == epochs:
            validation = predict(xy[validation_points], times[split["validation_times"]])
            validation_loss = errors(validation, water[np.ix_(split["validation_times"], validation_points)])["rmse_mm"]
            if validation_loss < best_loss:
                best_loss, best_state, best_epoch = validation_loss, copy.deepcopy(model.state_dict()), epoch
            history.append({"epoch": epoch, "training_mse_normalized": float(loss.detach()),
                            "validation_rmse_mm": validation_loss})
            print(f"Measured MLP epoch {epoch}: validation RMSE {validation_loss:.3f} mm", flush=True)
    model.load_state_dict(best_state)
    spatial, fallback = interpolate_space(xy, sensor_ids, water[np.ix_(training_times, sensor_ids)])
    # Both baselines see exactly the same sensors and training timestamps.
    linear = interp1d(times[training_times], spatial, axis=0)(times[split["test_times"]])
    neural = predict(xy, times[split["test_times"]])
    truth = water[split["test_times"]]
    report = {
        "task": "within-wave sparse WSE reconstruction with spatial and temporal holdouts",
        "processed_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "dataset": metadata, "seed": seed, "sensors": sensors,
        "training_frames": len(training_times), "validation_frames": len(split["validation_times"]),
        "test_frames": len(split["test_times"]), "heldout_spatial_points": len(heldout),
        "epochs": epochs, "selected_epoch": best_epoch, "training_seconds": time.monotonic() - start,
        "linear_interpolation": errors(linear[:, heldout], truth[:, heldout]),
        "supervised_mlp": errors(neural[:, heldout], truth[:, heldout]),
        "spatial_nearest_fallback_points": fallback, "history": history,
        "limitations": ["Within-wave interpolation, not future-event prediction or cross-event validation.",
                        "Reference data already include source filtering and smoothing.",
                        "Triangulation can span obstacle cavities; it is a simple baseline.",
                        "Only WSE is evaluated; depth and velocity are not measured targets."],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "lower": lower.tolist(), "scale": scale.tolist(),
                "target_mean": mean, "target_std": std, "seed": seed}, output / "model.pt")
    np.savez_compressed(output / "predictions.npz", xy_m=xy, time_s=times[split["test_times"]],
                        measured_wse_m=truth, linear_wse_m=linear, mlp_wse_m=neural)
    frame = len(truth) // 2
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), layout="constrained")
    values = [truth[frame], linear[frame], neural[frame], neural[frame] - truth[frame]]
    titles = ["Measured WSE", "Spatial + temporal interpolation", "Supervised MLP", "MLP residual"]
    bounds = np.percentile(truth[frame] * 1000, [1, 99])
    for ax, field, title in zip(axes.flat, values, titles):
        residual = title == "MLP residual"
        limits = [-max(np.max(np.abs(field)) * 1000, 1), max(np.max(np.abs(field)) * 1000, 1)] if residual else bounds
        artist = ax.scatter(xy[:, 0], xy[:, 1], c=field * 1000, s=7,
                            cmap="RdBu_r" if residual else "viridis", vmin=limits[0], vmax=limits[1])
        ax.scatter(xy[sensor_ids, 0], xy[sensor_ids, 1], s=5, c="black", alpha=0.25)
        ax.set(title=title, xlabel="x (m)", ylabel="y (m)", aspect="equal")
        fig.colorbar(artist, ax=ax, label="mm")
    fig.suptitle(f"{metadata['experiment']} wave {metadata['wave']} — held-out t = {times[split['test_times'][frame]]:.3f} s")
    fig.savefig(output / "reconstruction.png", dpi=180)
    plt.close(fig)
    return report


def run_transfer(source: Path, target: Path, trained: Path, output: Path) -> dict:
    """Evaluate a frozen model on another wave, without using its targets for fitting."""
    report = json.loads((trained / "metrics.json").read_text())
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    if report.get("processed_dataset_sha256") != fingerprint:
        raise ValueError("Source dataset differs from the training provenance; rerun measured-baseline")
    split = json.loads((trained / "split.json").read_text())
    checkpoint = torch.load(trained / "model.pt", map_location="cpu", weights_only=True)
    with np.load(source, allow_pickle=False) as data:
        source_xy, source_time, source_wse = data["xy"], data["time_s"], data["wse_m"]
        source_meta = json.loads(str(data["metadata"]))
    with np.load(target, allow_pickle=False) as data:
        xy, times, truth = data["xy"], data["time_s"], data["wse_m"]
        target_meta = json.loads(str(data["metadata"]))
    if source_meta["experiment"] != target_meta["experiment"] or source_meta["wave"] == target_meta["wave"]:
        raise ValueError("Transfer requires different waves of the same configuration")
    distance, source_indices = cKDTree(source_xy).query(xy)
    common = distance <= 1e-7
    excluded_spatial_points = int((~common).sum())
    if not common.any():
        raise ValueError("No matching spatial points between source and target grids")
    # Waves can have different missing pixels. Match coordinates; never align by row number.
    xy, truth = xy[common], truth[:, common]
    source_indices = source_indices[common]
    train_times, sensors = np.array(split["train_times"]), np.array(split["sensor_points"])
    keep = (times >= source_time[train_times].min()) & (times <= source_time[train_times].max())
    if not keep.any():
        raise ValueError("No target timestamps overlap the source observation interval")
    model = MLP(3, 1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    coordinates = make_coordinates(xy, times[keep])
    predictions = []
    torch.set_num_threads(2)
    with torch.no_grad():
        for chunk in np.array_split(coordinates, max(1, int(np.ceil(len(coordinates) / 8192)))):
            normalized = 2 * (chunk - np.array(checkpoint["lower"])) / np.array(checkpoint["scale"]) - 1
            prediction = model(torch.tensor(normalized, dtype=torch.float32)).numpy()[:, 0]
            predictions.append(prediction * checkpoint["target_std"] + checkpoint["target_mean"])
    neural = np.concatenate(predictions).reshape(int(keep.sum()), len(xy))
    spatial, _ = interpolate_space(source_xy, sensors, source_wse[np.ix_(train_times, sensors)])
    linear = interp1d(source_time[train_times], spatial[:, source_indices], axis=0)(times[keep])
    result = {
        "task": "Frozen within-configuration transfer; no target-wave observations used for fitting",
        "source": source_meta, "target": target_meta,
        "source_processed_sha256": fingerprint,
        "target_processed_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "evaluated_frames": int(keep.sum()), "excluded_frames_outside_source_time_range": int((~keep).sum()),
        "evaluated_points": len(xy), "evaluated_time_range_s": [float(times[keep][0]), float(times[keep][-1])],
        "excluded_target_points_without_source_coordinate_match": excluded_spatial_points,
        "frozen_source_interpolation": errors(linear, truth[keep]),
        "frozen_source_mlp": errors(neural, truth[keep]),
        "limitations": ["Same laboratory obstacle geometry; not independent geographic events.",
                        "Later wave starts with a changed bed; WSE remains relative to original bed.",
                        "One configuration and one trained seed; no Himalayan validation."],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
