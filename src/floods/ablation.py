"""Paired sensor-budget experiments using the showcase notebook's model code."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from .notebook import execute


def paired_sensor_split(points: int, sensors: int, maximum: int, seed: int):
    if not 3 <= sensors <= maximum < points:
        raise ValueError("Require 3 <= sensors <= maximum < spatial point count")
    permutation = np.random.default_rng(seed).permutation(points)
    return np.sort(permutation[:sensors]), np.sort(permutation[maximum:])


def build_run(source: Path, sensors: int, seed: int, steps: int, maximum: int, output: Path):
    if steps < 1 or not 3 <= sensors <= maximum:
        raise ValueError("Positive steps and 3 <= sensors <= maximum required")
    original = nbformat.read(source, as_version=4)
    cells = []
    found_config = found_split = False
    for cell in original.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("## 11."):
            break
        if cell.cell_type != "code" or "plt.show()" in cell.source or cell.source.startswith("df.info()"):
            continue
        copied = nbformat.v4.new_code_cell(cell.source)
        if 'DATA_SOURCE = "processed"' in copied.source:
            found_config = True
            copied.source += f"\nSEED = {seed}\nN_SENSORS = {sensors}\nMLP_STEPS = {steps}\n"
            copied.source += "random.seed(SEED)\nnp.random.seed(SEED)\ntorch.manual_seed(SEED)\n"
        if "sensor_ids = np.sort(np.random.default_rng(SEED).choice" in copied.source:
            found_split = True
            lines = copied.source.splitlines()
            lines = [line for line in lines if not line.startswith(("sensor_ids =", "heldout_ids ="))]
            insert = next(i for i, line in enumerate(lines) if line.startswith("train_rows ="))
            lines[insert:insert] = [
                "from floods.ablation import paired_sensor_split",
                f"sensor_ids, heldout_ids = paired_sensor_split(len(xy), N_SENSORS, {maximum}, SEED)",
            ]
            copied.source = "\n".join(lines)
        cells.append(copied)
    if not found_config or not found_split:
        raise ValueError("Notebook configuration/split changed; audit the experiment adapter")
    cells.append(nbformat.v4.new_code_cell(f"""
run_output = Path({str(output.resolve())!r})
run_output.mkdir(parents=True, exist_ok=True)
validation_results.to_csv(run_output / "validation.csv", index=False)
test_results.to_csv(run_output / "test.csv", index=False)
transfer_results.to_csv(run_output / "transfer.csv", index=False)
history_df.to_csv(run_output / "mlp_history.csv", index=False)
np.savez_compressed(run_output / "split.npz", sensors=sensor_ids, heldout=heldout_ids,
                    train_frames=train_frames, validation_frames=val_frames, test_frames=test_frames)
(run_output / "selection.json").write_text(json.dumps({{"selected_model": selected_model,
    "mlp_selected_step": best_step, "seed": SEED, "sensors": N_SENSORS, "steps": MLP_STEPS}}))
"""))
    return nbformat.v4.new_notebook(cells=cells, metadata=original.metadata)


def summarize(output: Path, reports: Path):
    rows = []
    for config in sorted(output.glob("sensors_*/seed_*/selection.json")):
        metadata = json.loads(config.read_text())
        for split in ("validation", "test", "transfer"):
            frame = pd.read_csv(config.parent / f"{split}.csv")
            frame["Sensors"], frame["Seed"], frame["Split"] = metadata["sensors"], metadata["seed"], split
            frame["Steps"] = metadata["steps"]
            frame["Selected by validation"] = frame["Model"] == metadata["selected_model"]
            rows.append(frame)
    runs = pd.concat(rows, ignore_index=True)
    summary = runs.groupby(["Split", "Sensors", "Model"], as_index=False).agg(
        rmse_mean_mm=("RMSE (mm)", "mean"), rmse_std_mm=("RMSE (mm)", "std"),
        mae_mean_mm=("MAE (mm)", "mean"), seeds=("Seed", "nunique"))
    reports.mkdir(parents=True, exist_ok=True)
    runs.to_csv(reports / "runs.csv", index=False)
    summary.to_csv(reports / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, split in zip(axes, ["test", "transfer"]):
        for name, frame in summary[summary.Split == split].groupby("Model"):
            ax.errorbar(frame.Sensors, frame.rmse_mean_mm, yerr=frame.rmse_std_mm.fillna(0),
                        marker="o", capsize=3, label=name)
        ax.set(title="Wave 1 holdout" if split == "test" else "Frozen transfer to wave 3",
               xlabel="Virtual sensor locations", ylabel="WSE RMSE (mm)", xticks=sorted(runs.Sensors.unique()))
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(reports / "sensor_ablation.png", dpi=180)
    plt.close(fig)
    return summary


def run_ablation(source: Path, output: Path, reports: Path, budgets, seeds, steps: int):
    budgets = sorted(set(budgets))
    seeds = list(dict.fromkeys(seeds))
    if not budgets or not seeds or min(budgets) < 3 or steps < 1:
        raise ValueError("Provide sensor budgets >= 3, seeds, and positive training steps")
    root = source.resolve().parent.parent
    experiment = {
        "notebook_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset_sha256": {name: hashlib.sha256((root / "reports/artifacts" / name).read_bytes()).hexdigest()
                           for name in ("O4_wave1.npz", "O4_wave3.npz")},
        "sensor_budgets": budgets, "seeds": seeds, "mlp_steps": steps,
        "design": "Nested sensors; fixed validation/test locations across budgets within each seed",
        "holdout_rule": "First max(budgets) permuted locations form candidate pool; all remaining locations held out",
        "variation": "Seed varies sensor placement and neural initialization; same two laboratory waves",
        "reference": "Separate paired experiment; its permutation differs from the showcase's random choice",
    }
    output.mkdir(parents=True, exist_ok=True)
    previous_protocol = output / "protocol.json"
    if previous_protocol.exists() and json.loads(previous_protocol.read_text()) != experiment:
        raise ValueError("Output contains a different protocol; choose a separate --output directory")
    expected = {output / f"sensors_{budget}" / f"seed_{seed}" / "selection.json"
                for budget in budgets for seed in seeds}
    if any(path not in expected for path in output.glob("sensors_*/seed_*/selection.json")):
        raise ValueError("Output contains other runs; choose a separate --output directory")
    previous_protocol.write_text(json.dumps(experiment, indent=2) + "\n")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "protocol.json").write_text(json.dumps(experiment, indent=2) + "\n")
    for seed in seeds:
        for sensors in budgets:
            run_dir = output / f"sensors_{sensors}" / f"seed_{seed}"
            print(f"Running {sensors} sensors, seed {seed}, {steps} MLP steps", flush=True)
            notebook = build_run(source, sensors, seed, steps, max(budgets), run_dir)
            execute(notebook, root, run_dir / "executed.ipynb")
    summary = summarize(output, reports)
    print(summary[summary.Split == "test"].to_string(index=False), flush=True)
    return summary
