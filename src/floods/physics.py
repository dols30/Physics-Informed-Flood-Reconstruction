"""Flat, fixed-bed, frictionless 1D shallow-water equations in dimensionless units."""

import numpy as np
import torch


def initial_height(x):
    """Smoothed wet-bed release: h_L=1, h_R=0.1; not a real disaster measurement."""
    if isinstance(x, torch.Tensor):
        return 0.1 + 0.9 * 0.5 * (1 - torch.tanh((x - 0.45) / 0.02))
    return 0.1 + 0.9 * 0.5 * (1 - np.tanh((x - 0.45) / 0.02))


def rusanov_flux(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Conservative local Lax-Friedrichs flux for state [h,q], g*=1."""
    def flux(state):
        h, q = state[:, 0], state[:, 1]
        return np.column_stack([q, q * q / h + 0.5 * h * h])
    speed = np.maximum(np.abs(left[:, 1] / left[:, 0]) + np.sqrt(left[:, 0]),
                       np.abs(right[:, 1] / right[:, 0]) + np.sqrt(right[:, 0]))
    return 0.5 * (flux(left) + flux(right)) - 0.5 * speed[:, None] * (right - left)


def solve_reference(cells: int = 128, frames: int = 41, final_time: float = 0.35) -> dict:
    if cells < 8 or frames < 3 or final_time <= 0:
        raise ValueError("Need at least 8 cells, 3 frames, and positive duration")
    dx = 1 / cells
    x = (np.arange(cells) + 0.5) * dx
    state = np.column_stack([initial_height(x), np.zeros(cells)])
    initial_mass = float(state[:, 0].sum() * dx)
    times = np.linspace(0, final_time, frames)
    history = [state.copy()]
    time_s, net_mass_inflow, steps = 0.0, 0.0, 0
    for target in times[1:]:
        while time_s < target - 1e-14:
            speed = np.max(np.abs(state[:, 1] / state[:, 0]) + np.sqrt(state[:, 0]))
            dt = min(0.4 * dx / speed, target - time_s)
            # Transmissive ghost cells; boundary influx is included in mass diagnostics.
            padded = np.vstack([state[0], state, state[-1]])
            interface_flux = rusanov_flux(padded[:-1], padded[1:])
            state = state - dt / dx * np.diff(interface_flux, axis=0)
            if not np.all(np.isfinite(state)) or np.any(state[:, 0] <= 0):
                raise FloatingPointError("Reference solver lost positive finite wet-bed height")
            net_mass_inflow += dt * (interface_flux[0, 0] - interface_flux[-1, 0])
            time_s += dt
            steps += 1
        history.append(state.copy())
    solution = np.stack(history)
    return {"x": x, "time": times, "state": solution,
            "diagnostics": {"cells": cells, "frames": frames, "steps": steps, "cfl": 0.4,
                            "min_height": float(solution[:, :, 0].min()),
                            "mass_balance_error": float(state[:, 0].sum() * dx - initial_mass - net_mass_inflow)}}


def shallow_water_residual(model: torch.nn.Module, xt: torch.Tensor) -> torch.Tensor:
    """Strong-form residuals; autograd differentiates physical dimensionless coordinates."""
    xt = xt.requires_grad_(True)
    state = model(xt)
    h, q = state[:, 0:1], state[:, 1:2]
    momentum_flux = q ** 2 / h + 0.5 * h ** 2

    def gradient(value):
        return torch.autograd.grad(value.sum(), xt, create_graph=True)[0]

    dh, dq, df = gradient(h), gradient(q), gradient(momentum_flux)
    return torch.stack([dh[:, 1] + dq[:, 0], dq[:, 1] + df[:, 0]], dim=1)
