import math

import torch
from torch import nn


class SpectralActivation(nn.Module):
    def __init__(self, harmonics=32, init_scale=1e-3, residual=True):
        super().__init__()
        self.harmonics = int(harmonics)
        self.residual = bool(residual)
        self.coeffs = nn.Parameter(init_scale * torch.randn(self.harmonics))

    def forward(self, x):
        y = 0.0
        for i in range(self.harmonics):
            y = y + self.coeffs[i] * torch.sin(2.0 * math.pi * (i + 1) * x)
        return x + y if self.residual else y


class LSAImageMLP(nn.Module):
    def __init__(self, width=256, hidden_layers=3, harmonics=32, residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        self.layers.append(nn.Linear(2, width))
        self.activations.append(SpectralActivation(harmonics, residual=residual))

        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(width, width))
            self.activations.append(SpectralActivation(harmonics, residual=residual))

        self.final = nn.Linear(width, 3)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            first = self.layers[0]
            first.weight.uniform_(-1.0 / first.in_features, 1.0 / first.in_features)
            first.bias.uniform_(-1.0 / first.in_features, 1.0 / first.in_features)

            for layer in self.layers[1:]:
                bound = math.sqrt(6.0 / layer.in_features) / 30.0
                layer.weight.uniform_(-bound, bound)
                layer.bias.uniform_(-bound, bound)

            bound = math.sqrt(6.0 / self.final.in_features) / 30.0
            self.final.weight.uniform_(-bound, bound)
            self.final.bias.uniform_(-bound, bound)

    def forward(self, x):
        h = x
        for layer, activation in zip(self.layers, self.activations):
            h = activation(layer(h))
        return self.final(h)
