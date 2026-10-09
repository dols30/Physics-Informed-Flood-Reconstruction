"""Read measured WSE without inventing depth, timestamps, or bed dynamics."""

import json
from pathlib import Path

import h5py
import numpy as np

from .download import RECORD_ID

MM_TO_M = 0.001


def experiment_path(data_dir: Path, experiment: str) -> Path:
    directory = data_dir / "raw" / f"zaragoza-{RECORD_ID}" / "extracted" / experiment
    if not directory.is_dir():
        raise FileNotFoundError(f"Download first: floods download --experiment {experiment}")
    return directory


def timestamp(path: Path) -> float:
    with h5py.File(path, "r") as handle:
        if "time_stamp[s]" not in handle.attrs:
            raise ValueError(f"Missing measured timestamp: {path}")
        value = float(handle.attrs["time_stamp[s]"])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite measured timestamp: {path}")
    return value


def water_files(directory: Path, wave: int) -> list[Path]:
    files = list((directory / f"water{wave}_h5").glob("*.h5"))
    if not files:
        raise ValueError(f"No water measurements for wave {wave} in {directory}")
    return sorted(files, key=timestamp)


def audit(data_dir: Path, experiment: str, output: Path) -> dict:
    directory = experiment_path(data_dir, experiment)
    entries = []
    for path in sorted(directory.rglob("*.h5")):
        with h5py.File(path, "r") as handle:
            entry = {"file": str(path.relative_to(directory)), "datasets": {}}
            for key in handle:
                dataset = handle[key]
                info = {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
                if dataset.ndim == 2 and np.issubdtype(dataset.dtype, np.number):
                    array = dataset[()]
                    finite = array[np.isfinite(array)]
                    info["finite_fraction"] = float(finite.size / array.size)
                    if finite.size:
                        info["range_native_units"] = [float(finite.min()), float(finite.max())]
                    if key == "mask":
                        info["unique_values"] = np.unique(array).tolist()
                entry["datasets"][key] = info
            if "time_stamp[s]" in handle.attrs:
                entry["time_s"] = float(handle.attrs["time_stamp[s]"])
            entries.append(entry)
    waves = {}
    for wave in (1, 2, 3):
        if not (directory / f"water{wave}_h5").is_dir():
            waves[str(wave)] = {"available": False, "snapshots": 0}
            continue
        times = np.array([timestamp(p) for p in water_files(directory, wave)])
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"Wave {wave} timestamps are not unique and increasing")
        waves[str(wave)] = {
            "available": True, "snapshots": len(times), "time_range_s": [float(times[0]), float(times[-1])],
            "median_interval_s": float(np.median(np.diff(times))),
        }
    report = {
        "dataset": f"https://zenodo.org/records/{RECORD_ID}", "experiment": experiment,
        "coordinates_and_elevations_native_unit": "mm", "time_native_unit": "s",
        "target": "water surface elevation relative to initial bed; not evolving-bed water depth",
        "velocity_observations": False, "during_wave_bed_observations": False,
        "waves": waves, "files": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def prepare(data_dir: Path, experiment: str, wave: int, stride: int, output: Path) -> dict:
    if stride < 1:
        raise ValueError("Grid stride must be at least one")
    directory = experiment_path(data_dir, experiment)
    selection = np.s_[::stride, ::stride]
    with h5py.File(directory / "offset.h5", "r") as handle:
        x = handle["gX"][()][selection].astype(float) * MM_TO_M
        y = handle["gY"][()][selection].astype(float) * MM_TO_M
        mask = handle["mask"][()][selection] != 0
    fields, times = [], []
    for path in water_files(directory, wave):
        with h5py.File(path, "r") as handle:
            water_x = handle["gX"][()][selection].astype(float) * MM_TO_M
            water_y = handle["gY"][()][selection].astype(float) * MM_TO_M
            if not np.allclose(x, water_x, atol=1e-7) or not np.allclose(y, water_y, atol=1e-7):
                raise ValueError(f"Coordinate grid differs from reference in {path}")
            fields.append(handle["wse[mm]"][()][selection].astype(float) * MM_TO_M)
            times.append(float(handle.attrs["time_stamp[s]"]))
    fields = np.stack(fields)
    # Strict intersection avoids filling missing measurements or reporting fabricated values.
    valid = mask & np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(fields), axis=0)
    if not np.any(valid):
        raise ValueError("No valid common measurement area after masking")
    metadata = {
        "experiment": experiment, "wave": wave, "grid_stride": stride,
        "spatial_points": int(valid.sum()), "frames": len(times),
        "units": {"xy": "m", "time": "s", "wse": "m"},
        "target": "WSE relative to initial bed", "source": f"https://zenodo.org/records/{RECORD_ID}",
        "mask_rule": "reference mask nonzero and finite across all selected frames",
        "excluded_grid_points": int(valid.size - valid.sum()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, xy=np.column_stack([x[valid], y[valid]]),
                        time_s=np.array(times), wse_m=fields[:, valid],
                        metadata=json.dumps(metadata))
    return metadata
