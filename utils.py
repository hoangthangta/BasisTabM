import torch
from prettytable import PrettyTable

import random
import numpy as np

import math
import torch.nn as nn
import torch.nn.functional as F

def cal_grad_mean(model):
    grad_mean = torch.mean(torch.stack([p.grad.abs().mean() for p in model.parameters() if p.grad is not None]))
    return grad_mean

def cal_grad_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    return total_norm ** 0.5  # L2 norm

def count_params(model, display = True):
    """
    Count the model's parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if (display == True): print(table)

    unused_params, unused_param_count = count_unused_params(model)

    used_params = max(0, total_params - unused_param_count)  # Prevent negative used params
    
    if (display == False): return total_params
    if unused_param_count > 0:
        print("Unused Parameters:", unused_params)
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Number of Unused Parameters: {unused_param_count}")
        print(f"Total Number of Used Parameters: {used_params}")
    else:
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Number of Used Parameters: {used_params}")

    return used_params
   

def count_unused_params(model):
    """
    Detect and count unused parameters
    """
    unused_params = []
    unused_param_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:  # Only count trainable params without gradients
            unused_params.append(name)
            unused_param_count += param.numel()

    return unused_params, unused_param_count

def remove_unused_params(model):
    """
    Remove unused parameters from the trained model
    """
    unused_params, _ = count_unused_params(model)
    for name in unused_params:
        if hasattr(model, name):
            delattr(model, name)  # Dynamically remove the unused layer
    return model
    

def set_seed(seed: int = 42, device="cuda"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

 
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return self.scale * x_norm

        
class MMNorm(nn.Module):
    def __init__(self, grid_min=-2.0, grid_max=2.0):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.eps = 1e-8

    def forward(self, x):
        x_norm = (x - x.min()) / (x.max() - x.min() + self.eps)
        x_norm = x_norm * (self.grid_max - self.grid_min) + self.grid_min
        return x_norm

def get_activation(name):
    funcs = {
        "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
        "elu": nn.ELU, "selu": nn.SELU, "tanh": nn.Tanh,
        "softplus": nn.Softplus, "identity": nn.Identity, "none": nn.Identity,
    }
    name = name.lower()
    if name not in funcs:
        raise ValueError(f"Unknown activation: {name}")
    return funcs[name]()