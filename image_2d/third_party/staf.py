import math

import torch
from torch import nn


class StafLayer(nn.Module):
    def __init__(self, in_features, out_features, tau, skip_conn=False, bias=True, is_first=False, omega_0=30, scale=10.0, init_weights=True):
        super().__init__()
        self.tau = tau
        self.omega_0 = omega_0
        self.is_first = is_first
        self.skip_conn = skip_conn
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_params()

    def init_params(self):
        self.ws = nn.Parameter(self.omega_0 * torch.rand(self.tau), requires_grad=True)
        self.phis = nn.Parameter(-math.pi + 2 * math.pi * torch.rand(self.tau), requires_grad=True)
        diversity_y = 1 / (2 * self.tau)
        samples = torch.distributions.Laplace(0, diversity_y).sample((self.tau,))
        self.bs = nn.Parameter(torch.sign(samples) * torch.sqrt(torch.abs(samples)), requires_grad=True)

    def forward(self, x):
        linear = self.linear(x)
        activated = self.param_act(linear)
        if self.is_first or not self.skip_conn:
            return activated
        return activated + linear

    def param_act(self, x):
        return (self.bs * torch.sin(self.ws * x.unsqueeze(-1) + self.phis)).sum(dim=-1)


class STAF(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, first_omega_0=30, hidden_omega_0=30.0, scale=10.0, tau=5, skip_conn=False):
        super().__init__()
        layers = [StafLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, scale=scale, tau=tau)]
        for _ in range(hidden_layers):
            layers.append(StafLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, scale=scale, tau=tau, skip_conn=skip_conn))
        if outermost_linear:
            layers.append(nn.Linear(hidden_features, out_features, dtype=torch.float))
        else:
            layers.append(StafLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0, scale=scale, tau=tau, skip_conn=skip_conn))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
