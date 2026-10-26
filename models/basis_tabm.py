import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class BasisFunction(nn.Module):
    def __init__(self, basis="rbf", num_grids=8, grid_min=-2.0, grid_max=2.0, use_width=False, width=1.0):
        super().__init__()
        self.basis, self.num_grids = basis.lower(), num_grids
        self.grid_min, self.grid_max, self.use_width = grid_min, grid_max, use_width
        if self.basis not in {"rbf", "sech", "laplace", "cauchy", "mexican_hat", "gaussian_d2"}:
            raise ValueError(f"Unknown basis: {basis}")
        if width <= 0:
            raise ValueError("width must be positive")
        self.grid = nn.Parameter(torch.empty(num_grids))
        if use_width:
            self.log_width = nn.Parameter(torch.empty(()))
        self.reset_parameters(width)

    @torch.no_grad()
    def reset_parameters(self, width=1.0):
        self.grid.copy_(torch.linspace(self.grid_min, self.grid_max, self.num_grids, device=self.grid.device, dtype=self.grid.dtype))
        if self.use_width:
            self.log_width.fill_(math.log(width))

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
            return 1 / (1 + z2)
        e = torch.exp(-0.5 * z2)
        if self.basis == "rbf":
            return e
        if self.basis == "mexican_hat":
            return (1 - z2) * e
        return (z2 - 1) * e


class EnsembleBatchNorm1d(nn.Module):
    """BatchNorm on (B,k,d), with separate statistics for each member."""
    def __init__(self, dim, k):
        super().__init__()
        self.bn = nn.BatchNorm1d(k * dim)

    def reset_parameters(self):
        self.bn.reset_parameters()

    def forward(self, x):
        return self.bn(x.flatten(-2)).unflatten(-1, x.shape[-2:])


def _random_signs(*shape):
    """+1/-1 entries with equal probability (TabM / BatchEnsemble adapter init)."""
    return torch.randint(0, 2, shape).float().mul_(2).sub_(1)


class FeatureBasis(nn.Module):
    """Learnable univariate function per input feature (KAN-style, applied once at the input):

        u_i = x_i + sum_g c[i, g] * phi_g(x_i)

    c starts at zero, so the model starts as plain TabM and learns where a per-feature
    non-linearity helps. Cost: in_dim * num_grids (+ num_grids grid centres).
    """
    def __init__(self, in_dim, num_grids=8, basis="rbf", grid_min=-2.0, grid_max=2.0,
                 use_width=False, width=1.0):
        super().__init__()
        self.basis = BasisFunction(basis, num_grids, grid_min, grid_max, use_width, width)
        self.coef = nn.Parameter(torch.zeros(in_dim, num_grids))

    def forward(self, x):                                   # x: (..., in_dim)
        return x + (self.basis(x) * self.coef).sum(-1)


class BasisTabM_Layer(nn.Module):
    """Shared linear weights with member-specific scaling and basis coefficients.

    basis_mode    : "none" -> plain BatchEnsemble/TabM layer (default; the basis is used by FeatureBasis).
                    "mix"  -> old design: h = (1-alpha)*h_tabm + alpha*W(phi(x)), phi aggregated
                              over the grid by `basis_agg` ("sum" = v1, "proj" = v2).
    ensemble_init : "random_signs" (default) or "ones" (old behaviour: all k members stay identical).
    alpha_mode    : "member" (k, out) or "shared" (1, out); only for basis_mode="mix".
    """
    def __init__(
        self, in_dim, out_dim, k=8, basis="rbf",
        grid_min=-2.0, grid_max=2.0, use_width=False, width=1.0,
        norm1_type="layer", norm2_type="layer", activation="silu",
        apply_norm=True, dropout=0.0,
        num_grids=None, ensemble_init="random_signs", alpha_mode="member", basis_agg="sum", basis_mode="none",
    ):
        super().__init__()
        if ensemble_init not in ("random_signs", "ones"):
            raise ValueError(f"Unknown ensemble_init: {ensemble_init}")
        if alpha_mode not in ("member", "shared"):
            raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
        if basis_agg not in ("sum", "proj"):
            raise ValueError(f"Unknown basis_agg: {basis_agg}")
        if basis_mode not in ("mix", "none"):
            raise ValueError(f"Unknown basis_mode: {basis_mode}")
        num_grids = k if num_grids is None else num_grids
        self.in_dim, self.out_dim, self.k = in_dim, out_dim, k
        self.basis_agg, self.basis_mode = basis_agg, basis_mode
        self.shared_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.shared_weight)

        if ensemble_init == "random_signs":
            self.ensemble_input = nn.Parameter(_random_signs(k, in_dim))
            self.ensemble_output = nn.Parameter(_random_signs(k, out_dim))
        else:
            self.ensemble_input = nn.Parameter(torch.ones(k, in_dim))
            self.ensemble_output = nn.Parameter(torch.ones(k, out_dim))
        self.ensemble_bias = nn.Parameter(torch.zeros(k, out_dim))

        if basis_mode == "mix":
            self.basis = BasisFunction(basis, num_grids, grid_min, grid_max, use_width, width)
            self.alpha = nn.Parameter(torch.zeros(k if alpha_mode == "member" else 1, out_dim))
            if basis_agg == "proj":
                self.basis_proj_weight = nn.Parameter(torch.empty(1, num_grids))
                nn.init.xavier_uniform_(self.basis_proj_weight)
                self.basis_proj_bias = nn.Parameter(torch.zeros(1))

        self.norm1 = self._make_norm(norm1_type, in_dim, k)
        self.norm2 = self._make_norm(norm2_type, out_dim, k)
        self.apply_norm = apply_norm
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_norm(norm_type, dim, k):
        norm_type = norm_type.lower()
        if norm_type == "layer":
            return nn.LayerNorm(dim)
        if norm_type == "rms":
            return RMSNorm(dim)
        if norm_type == "batch":
            return EnsembleBatchNorm1d(dim, k)
        if norm_type in ("none", "identity", ""):
            return nn.Identity()
        raise ValueError(f"Unknown normalization: {norm_type}")

    def forward(self, x, activate=True):
        if x.ndim != 3 or x.shape[1:] != (self.k, self.in_dim):
            raise ValueError(f"Expected input (B,{self.k},{self.in_dim}), got {tuple(x.shape)}")

        if self.apply_norm:
            x = self.norm1(x)
        h = F.linear(x * self.ensemble_input, self.shared_weight)
        h = h * self.ensemble_output + self.ensemble_bias

        if self.basis_mode == "mix":
            phi = self.basis(x)                                # (B, k, in_dim, num_grids)
            if self.basis_agg == "sum":
                phi = phi.sum(dim=-1)
            else:
                phi = F.linear(phi, self.basis_proj_weight, self.basis_proj_bias).squeeze(-1)
            h_basis = F.linear(phi, self.shared_weight)
            h = (1 - self.alpha) * h + self.alpha * h_basis

        if self.apply_norm:
            h = self.norm2(h)
        if activate:
            h = self.dropout(self.activation(h))
        return h


class BasisTabM(nn.Module):
    """Input (B,in) or (B,k,in) -> member outputs (B,k,out).

    Default = the "FB" design: TabM-style random-sign adapters (members differ from the start),
    plain BatchEnsemble layers, and a learnable per-feature RBF transform at the input.
    The earlier v1 model is: ensemble_init="ones", basis_mode="mix", feature_basis=False.
    """
    def __init__(
        self, net_layers, k=8, basis="rbf",
        grid_min=-2.0, grid_max=2.0, use_width=False, width=1.0,
        norm1_type="layer", norm2_type="layer", activation="silu",
        dropout=0.0,
        num_grids=None, ensemble_init="random_signs", alpha_mode="member", basis_agg="sum", basis_mode="none",
        feature_basis=True, feature_grids=8,
    ):
        super().__init__()
        if len(net_layers) < 2:
            raise ValueError("net_layers must contain at least input and output dimensions")
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k
        n = len(net_layers) - 1
        self.feature_basis = (FeatureBasis(net_layers[0], feature_grids, basis, grid_min, grid_max,
                                           use_width, width) if feature_basis else None)
        self.layers = nn.ModuleList([
            BasisTabM_Layer(
                net_layers[i], net_layers[i + 1], k=k, basis=basis,
                grid_min=grid_min, grid_max=grid_max, use_width=use_width, width=width,
                norm1_type=norm1_type, norm2_type=norm2_type, activation=activation,
                apply_norm=(i < n - 1),
                dropout=dropout if i < n - 1 else 0.0,
                num_grids=num_grids, ensemble_init=ensemble_init,
                alpha_mode=alpha_mode, basis_agg=basis_agg, basis_mode=basis_mode,
            )
            for i in range(n)
        ])

    def forward(self, x, return_members=True, task="classification"):
        if self.feature_basis is not None and x.ndim == 2:
            x = self.feature_basis(x)
        if x.ndim == 2:
            x = x.unsqueeze(1).expand(-1, self.k, -1)
        elif self.feature_basis is not None:
            x = self.feature_basis(x)
        if x.ndim != 3 or x.shape[1] != self.k:
            raise ValueError(f"Expected input (B,in) or (B,{self.k},in), got {tuple(x.shape)}")

        for i, layer in enumerate(self.layers):
            x = layer(x, activate=(i < len(self.layers) - 1))

        if return_members:
            return x
        if task == "classification":
            return torch.logsumexp(F.log_softmax(x, dim=-1), dim=1) - math.log(self.k)
        if task == "regression":
            return x.mean(dim=1)
        raise ValueError(f"Unknown task: {task}")

    @torch.no_grad()
    def predict(self, x, task="classification"):
        y = self.forward(x, return_members=True)
        if task == "classification":
            return y.softmax(dim=-1).mean(dim=1)
        if task == "regression":
            return y.mean(dim=1)
        raise ValueError(f"Unknown task: {task}")


def ensemble_loss(y, target, task="classification"):
    """Average the losses across ensemble members."""
    k = y.shape[1]
    if task == "classification":
        return F.cross_entropy(y.flatten(0, 1), target.repeat_interleave(k))
    if task == "regression":
        return F.mse_loss(y, target.unsqueeze(1).expand_as(y))
    raise ValueError(f"Unknown task: {task}")
