"""Controlled PINN versus data-only ablation; all observations here are simulated."""

import copy
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import ShallowWaterNet
from .physics import solve_reference, shallow_water_residual


def run_synthetic(output: Path, epochs: int = 1500, seed: int = 42, physics_weight: float = 0.01) -> dict:
    if epochs < 1 or physics_weight <= 0:
        raise ValueError("Epoch count and physics weight must be positive")
    output.mkdir(parents=True, exist_ok=True)
    reference = solve_reference()
    refined = solve_reference(cells=256)
    x, times, states = reference["x"], reference["time"], reference["state"]
    delta = np.interp(x, refined["x"], refined["state"][-1, :, 0]) - states[-1, :, 0]
    reference["diagnostics"]["final_height_rmse_against_256_cells"] = float(np.sqrt(np.mean(delta ** 2)))
    np.savez_compressed(output / "reference.npz", x=x, time=times, h=states[:, :, 0], q=states[:, :, 1])
    sensors = np.linspace(6, len(x) - 7, 12, dtype=int)
    heldout = np.setdiff1d(np.arange(len(x)), sensors)
    frames = np.arange(len(times))
    interior = (frames > 0) & (frames < len(times) - 1)
    val = frames[interior & (frames % 5 == 2)]
    test = frames[interior & (frames % 5 == 4)]
    train = np.setdiff1d(frames, np.concatenate([val, test]))
    # Initial and boundary h,q are known synthetic conditions, shared by both models.
    def coordinates(xs, ts):
        return torch.tensor(np.column_stack([np.tile(xs, len(ts)), np.repeat(ts, len(xs))]), dtype=torch.float32)
    train_xt = coordinates(x[sensors], times[train])
    train_h = torch.tensor(states[np.ix_(train, sensors)][:, :, 0].reshape(-1, 1), dtype=torch.float32)
    initial_xt = coordinates(x, [0])
    initial_hq = torch.tensor(states[0], dtype=torch.float32)
    boundary_xt = coordinates(x[[0, -1]], times)
    boundary_hq = torch.tensor(states[:, [0, -1]].reshape(-1, 2), dtype=torch.float32)
    val_xt = coordinates(x[heldout], times[val])
    val_h = states[np.ix_(val, heldout)][:, :, 0].reshape(-1)
    test_xt = coordinates(x[heldout], times[test])
    test_hq = states[np.ix_(test, heldout)].reshape(-1, 2)
    torch.set_num_threads(2)
    results, models, histories = {}, {}, {}
    start = time.monotonic()
    for name, weight in [("data_only", 0.0), ("pinn", physics_weight)]:
        torch.manual_seed(seed)
        # Separate generators ensure minibatches and initial weights match in the ablation.
        batches = torch.Generator().manual_seed(seed + 1)
        collocation = torch.Generator().manual_seed(seed + 2)
        model = ShallowWaterNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_loss, best_state, best_epoch = float("inf"), None, 0
        history = []
        for epoch in range(1, epochs + 1):
            ids = torch.randint(len(train_xt), (128,), generator=batches)
            optimizer.zero_grad()
            observation_loss = torch.mean((model(train_xt[ids])[:, :1] - train_h[ids]) ** 2)
            condition_loss = torch.mean((model(initial_xt) - initial_hq) ** 2) + torch.mean((model(boundary_xt) - boundary_hq) ** 2)
            loss = observation_loss + condition_loss
            if weight:
                points = torch.rand((128, 2), generator=collocation)
                points[:, 0] = x[0] + points[:, 0] * (x[-1] - x[0])
                points[:, 1] *= times[-1]
                pde_loss = torch.mean(shallow_water_residual(model, points) ** 2)
                loss = loss + weight * min(1, epoch / 200) * pde_loss
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0 or epoch == epochs:
                with torch.no_grad():
                    predicted = model(val_xt).numpy()[:, 0]
                rmse = float(np.sqrt(np.mean((predicted - val_h) ** 2)))
                if rmse < best_loss:
                    best_loss, best_state, best_epoch = rmse, copy.deepcopy(model.state_dict()), epoch
                history.append({"epoch": epoch, "validation_height_rmse": rmse})
                print(f"Synthetic {name} epoch {epoch}: validation height RMSE {rmse:.4f}", flush=True)
        model.load_state_dict(best_state)
        with torch.no_grad():
            prediction = model(test_xt).numpy()
        diagnostic_generator = torch.Generator().manual_seed(seed + 3)
        diagnostic_points = torch.rand((1024, 2), generator=diagnostic_generator)
        diagnostic_points[:, 0] = x[0] + diagnostic_points[:, 0] * (x[-1] - x[0])
        diagnostic_points[:, 1] *= times[-1]
        residual = shallow_water_residual(model, diagnostic_points).detach().numpy()
        results[name] = {
            "selected_epoch": best_epoch, "physics_weight": weight,
            "heldout_height_rmse_dimensionless": float(np.sqrt(np.mean((prediction[:, 0] - test_hq[:, 0]) ** 2))),
            "heldout_discharge_rmse_dimensionless": float(np.sqrt(np.mean((prediction[:, 1] - test_hq[:, 1]) ** 2))),
            "continuity_residual_rms": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
            "momentum_residual_rms": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        }
        torch.save({"state_dict": model.state_dict(), "seed": seed, "physics_weight": weight}, output / f"{name}.pt")
        models[name], histories[name] = model, history
    report = {
        "data_origin": "numerical Rusanov simulation, not measured or Nepal event data",
        "physics": "dimensionless 1D flat fixed-bed frictionless shallow-water equations, g*=1",
        "seed": seed, "epochs": epochs, "sensors": sensors.tolist(),
        "train_times": train.tolist(), "validation_times": val.tolist(), "test_times": test.tolist(),
        "initial_and_boundary_conditions": "Known synthetic h,q, supplied equally to both models",
        "reference_solver": reference["diagnostics"], "training_seconds": time.monotonic() - start,
        "results": results, "history": histories,
        "limitations": ["One seed and one synthetic case; no research superiority claim.",
                        "Strong-form PINNs smooth shocks; weak or finite-volume losses may be needed.",
                        "Reference is a first-order numerical approximation with resolution sensitivity.",
                        "Not the erodible-bed laboratory physics; no Nepal transfer has been tested."],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained", sharey=True)
    plot_frames = test[[0, len(test) // 2, -1]]
    predictions = {}
    with torch.no_grad():
        for name, model in models.items():
            predictions[name] = model(coordinates(x, times[test])).numpy().reshape(len(test), len(x), 2)
    np.savez_compressed(output / "predictions.npz", x=x, time=times[test], reference=states[test], **predictions)
    for ax, frame in zip(axes, plot_frames):
        ax.plot(x, states[frame, :, 0], c="black", lw=2, label="Numerical reference")
        for name, model in models.items():
            with torch.no_grad():
                predicted = model(coordinates(x, times[[frame]])).numpy()[:, 0]
            ax.plot(x, predicted, label="PINN" if name == "pinn" else "Data-only NN")
        ax.scatter(x[sensors], states[frame, sensors, 0], s=12, c="black", marker="x", label="Sensor locations")
        ax.set(title=f"Held-out t* = {times[frame]:.3f}", xlabel="x*", ylabel="h*")
    axes[0].legend(fontsize=8)
    fig.suptitle("Simulated wet-bed release — controlled physics ablation")
    fig.savefig(output / "pinn_comparison.png", dpi=180)
    plt.close(fig)
    return report
