import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class BasisFunction(nn.Module):
    def __init__(self, basis="sech", num_grids=8, grid_min=-2.0, grid_max=2.0, use_width=False, width=1.0):
        super().__init__()
        self.basis, self.use_width = basis.lower(), use_width
        supported = {"rbf", "sech", "laplace", "cauchy", "mexican_hat", "gaussian_d2"}
        if self.basis not in supported:
            raise ValueError(f"Unknown basis: {basis}. Choose from {sorted(supported)}")
        self.grid = nn.Parameter(torch.linspace(grid_min, grid_max, num_grids))
        if use_width:
            if width <= 0:
                raise ValueError("width must be positive")
            self.log_width = nn.Parameter(torch.tensor(float(width)).log())

    def forward(self, x):
        z = x.unsqueeze(-1) - self.grid
        if self.use_width:
            z = z * torch.exp(-self.log_width).clamp(max=1e8)
        if self.basis == "sech":
            e = torch.exp(-z.abs())
            return 2 * e / (1 + e.square())
        if self.basis == "laplace":
            return torch.exp(-z.abs())
        z2 = z.square()
        if self.basis == "cauchy":
            return torch.reciprocal(1 + z2)
        e = torch.exp(-0.5 * z2)
        if self.basis == "rbf":
            return e
        if self.basis == "mexican_hat":
            return (1 - z2) * e
        return (z2 - 1) * e


class EnsembleBatchNorm1d(nn.Module):
    """BatchNorm on (B,k,d) with separate statistics for each of the k members."""

    def __init__(self, dim, k):
        super().__init__()
        self.bn = nn.BatchNorm1d(k * dim)

    def forward(self, x):                                   # (B,k,d)
        return self.bn(x.flatten(-2)).unflatten(-1, x.shape[-2:])


class BasisTabM_Layer(nn.Module):
    """One layer of k parallel members. Input (B,k,in) -> output (B,k,out).

    k is both the ensemble size and the number of grid points of the basis.
    Members are NEVER mixed inside the layer (principle 1: separate paths).
    """

    def __init__(
        self, in_dim, out_dim, k=8, basis="sech",
        grid_min=-2.0, grid_max=2.0, use_width=False, width=1.0,
        norm1_type="none", norm2_type="layer", activation="silu",
        apply_norm=True, dropout=0.0, first_layer=False, scaling_init="random-signs",
    ):
        super().__init__()
        if scaling_init not in ("random-signs", "normal"):
            raise ValueError("scaling_init must be 'random-signs' or 'normal'")
        self.in_dim, self.out_dim, self.k = in_dim, out_dim, k
        self.first_layer, self.scaling_init = first_layer, scaling_init

        # shared by all members
        self.shared_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.basis = BasisFunction(basis, k, grid_min, grid_max, use_width, width)  # k grid points

        # per member (BatchEnsemble adapters)
        self.ensemble_input = nn.Parameter(torch.empty(k, in_dim))    # r
        self.ensemble_output = nn.Parameter(torch.empty(k, out_dim))  # s
        self.ensemble_bias = nn.Parameter(torch.empty(k, out_dim))    # b

        # per member basis coefficients: member m combines the k grid bumps
        self.grid_coef = nn.Parameter(torch.empty(k, k))              # (member, grid)
        self.grid_bias = nn.Parameter(torch.empty(k))
        self.alpha = nn.Parameter(torch.full((k, out_dim), 0.1))      # weight of basis branch

        self.norm1 = self._make_norm(norm1_type, in_dim, k)
        self.norm2 = self._make_norm(norm2_type, out_dim, k)
        self.apply_norm = apply_norm
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)   # applied on (B,k,d): different mask per member
        self.reset_parameters()

    @staticmethod
    def _make_norm(norm_type, dim, k):
        norm_type = norm_type.lower()
        if norm_type == "layer": return nn.LayerNorm(dim)            # normalizes last dim
        if norm_type == "rms": return RMSNorm(dim)
        if norm_type == "batch": return EnsembleBatchNorm1d(dim, k)
        if norm_type in ("none", "identity", ""): return nn.Identity()
        raise ValueError(f"Unknown normalization: {norm_type}")

    @torch.no_grad()
    def reset_parameters(self):
        bound = self.in_dim ** -0.5
        nn.init.uniform_(self.shared_weight, -bound, bound)

        # TabM init: members differ ONLY by the first layer's input scaling
        if self.first_layer:
            if self.scaling_init == "random-signs":
                self.ensemble_input.bernoulli_(0.5).mul_(2).sub_(1)
            else:
                nn.init.normal_(self.ensemble_input)
        else:
            nn.init.ones_(self.ensemble_input)
        nn.init.ones_(self.ensemble_output)
        # same bias for all members at init
        self.ensemble_bias.copy_(torch.empty(self.out_dim).uniform_(-bound, bound))

        g = self.k ** -0.5
        nn.init.uniform_(self.grid_coef, -g, g)
        nn.init.uniform_(self.grid_bias, -g, g)

    def forward(self, x, activate=True):
        if x.ndim != 3 or x.shape[1:] != (self.k, self.in_dim):
            raise ValueError(f"Expected input (B,{self.k},{self.in_dim}), got {tuple(x.shape)}")

        if self.apply_norm:
            x = self.norm1(x)

        # TabM branch: ((x * r_m) W^T) * s_m + b_m                       -> (B,k,out)
        h_tabm = F.linear(x * self.ensemble_input, self.shared_weight)
        h_tabm = h_tabm * self.ensemble_output + self.ensemble_bias

        # Basis branch: member m uses its own coefficients over the k grid bumps
        phi = self.basis(x)                                              # (B,k,in,k_grid)
        g = torch.einsum("bmig,mg->bmi", phi, self.grid_coef) + self.grid_bias[:, None]
        h_basis = F.linear(g, self.shared_weight)                        # (B,k,out)

        # Combine branches 
        h = h_tabm + self.alpha * h_basis

        if self.apply_norm:
            h = self.norm2(h)
        if activate:
            h = self.dropout(self.activation(h))
        return h


class BasisTabM(nn.Module):
    """Input (B,input_dim) or (B,k,input_dim) -> logits (B,k,output_dim), one per member.

    Train with `ensemble_loss` (per-member loss); predict with `predict` (average).
    """

    def __init__(
        self, net_layers, k=8, basis="rbf",
        grid_min=-2.0, grid_max=2.0, use_width=False, width=1.0,
        norm1_type="layer", norm2_type="layer", activation="silu",
        dropout=0.0, scaling_init="random-signs",
    ):
        super().__init__()
        if len(net_layers) < 2:
            raise ValueError("net_layers must contain at least input and output dimensions")
        self.k = k
        n = len(net_layers) - 1
        self.layers = nn.ModuleList([
            BasisTabM_Layer(
                net_layers[i], net_layers[i + 1], k=k, basis=basis,
                grid_min=grid_min, grid_max=grid_max, use_width=use_width, width=width,
                norm1_type=norm1_type, norm2_type=norm2_type, activation=activation,
                apply_norm=(i < n - 1), dropout=dropout,
                first_layer=(i == 0), scaling_init=scaling_init,
            )
            for i in range(n)
        ])

    def forward(self, x, return_members=True, task="classification"):
        """
        return_members=True  -> (B,k,out): one prediction per member. Use for TRAINING
                                with `ensemble_loss`
        return_members=False -> (B,out): the ensemble prediction.
            classification: log of the averaged probabilities. It behaves like logits:
                softmax(out) == mean of member probabilities, argmax(out) == ensemble
                class, F.cross_entropy(out, y) == ensemble validation loss.
            regression: mean of the member outputs.
        """
        if x.ndim == 2:                                   # same batch for every member
            x = x.unsqueeze(1).expand(-1, self.k, -1)     # (B,k,in), no copy
        for i, layer in enumerate(self.layers):
            x = layer(x, activate=(i < len(self.layers) - 1))
        if return_members:
            return x                                      # (B,k,out)
        if task == "classification":
            return x.log_softmax(-1).logsumexp(1) - math.log(self.k)   # log mean prob
        return x.mean(1)

    @torch.no_grad()
    def predict(self, x, task="classification"):
        """Principle 3: average the k members' predictions."""
        y = self.forward(x)
        if task == "classification":
            return y.softmax(-1).mean(1)                  # (B,C) probabilities
        return y.mean(1)                                  # (B,out) regression


def ensemble_loss(y, target, task="classification"):
    """Principle 2: mean of the k per-member losses (NOT the loss of the mean)."""
    k = y.shape[1]
    if task == "classification":
        return F.cross_entropy(y.flatten(0, 1), target.repeat_interleave(k))
    return F.mse_loss(y, target.unsqueeze(1).expand_as(y))