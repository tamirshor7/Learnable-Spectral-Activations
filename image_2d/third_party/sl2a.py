import math

import torch
from torch import nn


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, init_method="xavier_uniform"):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.cheby_coeffs)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.cheby_coeffs, a=0, mode="fan_in", nonlinearity="relu")
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.cheby_coeffs, a=0, mode="fan_in", nonlinearity="relu")
        elif init_method == "orthogonal":
            nn.init.orthogonal_(self.cheby_coeffs)
        elif init_method == "uniform":
            nn.init.uniform_(self.cheby_coeffs, a=-0.5, b=0.5)
        elif init_method == "normal":
            nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        else:
            raise ValueError(init_method)
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        x = torch.tanh(x)
        x = x.view((-1, self.inputdim, 1)).expand(-1, -1, self.degree + 1)
        x = x.acos()
        x *= self.arange
        x = x.cos()
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        return y.view(-1, self.outdim)


class ChebyLayer(nn.Module):
    def __init__(self, in_features, out_features, deg, init_method):
        super().__init__()
        self.cheby = ChebyKANLayer(in_features, out_features, deg, init_method)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        return self.norm(self.cheby(x))


class LowRankReLULayer(nn.Module):
    def __init__(self, in_features, out_features, rank=32, bias=True, nonlinearity="relu", linear_init_type="kaiming_uniform"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.nonlinearity = nonlinearity
        self.weight_left = nn.Parameter(torch.Tensor(in_features, rank))
        self.weight_right = nn.Parameter(torch.Tensor(rank, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters(linear_init_type)

    def reset_parameters(self, init_type):
        if init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight_left, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_right, a=math.sqrt(5))
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(self.weight_left, a=math.sqrt(5))
            nn.init.kaiming_normal_(self.weight_right, a=math.sqrt(5))
        elif init_type == "orthogonal":
            nn.init.orthogonal_(self.weight_left)
            nn.init.orthogonal_(self.weight_right)
        elif init_type == "uniform":
            nn.init.uniform_(self.weight_left, a=-0.5, b=0.5)
            nn.init.uniform_(self.weight_right, a=-0.5, b=0.5)
        elif init_type == "normal":
            nn.init.normal_(self.weight_left, mean=0.0, std=1 / (self.in_features * self.rank))
            nn.init.normal_(self.weight_right, mean=0.0, std=1 / (self.rank * self.out_features))
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight_left)
            nn.init.xavier_uniform_(self.weight_right)
        else:
            raise ValueError(init_type)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_left)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = torch.matmul(self.weight_left, self.weight_right)
        output = torch.matmul(x, weight)
        if self.bias is not None:
            output += self.bias
        if self.nonlinearity == "relu":
            return nn.functional.relu(output)
        if self.nonlinearity in (None, "none"):
            return output
        raise ValueError(self.nonlinearity)


class SL2A(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, deg=512, outermost_linear=True, nonlinearity="relu", rank=32, init_method="xavier_uniform", linear_init_type="kaiming_uniform"):
        super().__init__()
        layers = [ChebyLayer(in_features, hidden_features, deg=deg, init_method=init_method)]
        final = nn.Linear(hidden_features, out_features)
        for _ in range(hidden_layers):
            layers.append(LowRankReLULayer(hidden_features, hidden_features, rank=rank, nonlinearity=nonlinearity, linear_init_type=linear_init_type))
        if not outermost_linear:
            raise NotImplementedError
        layers.append(final)
        self.net = nn.ModuleList(layers)

    def forward(self, coords):
        coords = coords.squeeze()
        for i, layer in enumerate(self.net):
            if i == 0:
                x = layer(coords)
                y = x
            else:
                y = layer(torch.einsum("ij,ij->ij", x, y))
        return y
