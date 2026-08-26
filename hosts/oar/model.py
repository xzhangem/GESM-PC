"""SIREN velocity field, optional 4-channel Softmax ADW head."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = np.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


def _siren_stack(in_features, hidden_features, hidden_layers, first_omega_0, hidden_omega_0):
    layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
    for _ in range(hidden_layers):
        layers.append(
            SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0)
        )
    return nn.Sequential(*layers)


class SirenV(nn.Module):
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=3,
                 first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        self.backbone = _siren_stack(
            in_features, hidden_features, hidden_layers, first_omega_0, hidden_omega_0
        )
        self.final = nn.Linear(hidden_features, 3)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_features) / hidden_omega_0
            self.final.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.final.bias)

    def forward(self, coords):
        return self.final(self.backbone(coords)), None


class SirenVW(nn.Module):
    def __init__(self, in_features=3, hidden_features=128, hidden_layers=3,
                 first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        self.backbone = _siren_stack(
            in_features, hidden_features, hidden_layers, first_omega_0, hidden_omega_0
        )
        self.final = nn.Linear(hidden_features, 7)
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_features) / hidden_omega_0
            self.final.weight.uniform_(-bound, bound)
            nn.init.zeros_(self.final.bias)

    def forward(self, coords):
        x = self.final(self.backbone(coords))
        return x[:, :3], torch.softmax(x[:, 3:], dim=-1)
