"""Small CPU-friendly networks shared by the two benchmark tracks."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, inputs: int, outputs: int, width: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.layers(coordinates)


class ShallowWaterNet(MLP):
    """Dimensionless h > 0 and unconstrained discharge q, for a wet-bed benchmark."""

    def __init__(self, width: int = 64):
        super().__init__(2, 2, width)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        raw = super().forward(torch.stack([2 * xt[:, 0] - 1, 2 * xt[:, 1] / 0.35 - 1], dim=1))
        height = torch.nn.functional.softplus(raw[:, :1]) + 1e-5
        return torch.cat([height, raw[:, 1:]], dim=1)
