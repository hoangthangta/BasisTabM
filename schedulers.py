import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class CustomOneCycleLR(_LRScheduler):
    def __init__(
        self, optimizer, max_lr, total_steps, pct_start=0.3,
        anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
        momentum_range=(0.85, 0.95), annihilation_frac=0.1, last_epoch=-1
    ):
        if max_lr <= 0 or total_steps < 1:
            raise ValueError("max_lr must be positive and total_steps >= 1.")
        if div_factor <= 0 or final_div_factor <= 0:
            raise ValueError("div_factor and final_div_factor must be positive.")
        if not 0 < pct_start < 1 or not 0 <= annihilation_frac < 1:
            raise ValueError("Invalid pct_start or annihilation_frac.")
        if anneal_strategy not in {"cos", "linear"}:
            raise ValueError("anneal_strategy must be 'cos' or 'linear'.")

        self.max_lr = float(max_lr)
        self.total_steps = int(total_steps)
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.momentum_range = momentum_range
        self.annihilation_steps = min(
            max(0, int(total_steps * annihilation_frac)),
            max(0, total_steps - 2),
        )
        self.step_size_up = min(
            max(1, int(total_steps * pct_start)),
            max(1, total_steps - self.annihilation_steps),
        )
        self.step_size_down = max(
            0, total_steps - self.step_size_up - self.annihilation_steps
        )

        self.initial_lrs = [
            max_lr / div_factor for _ in optimizer.param_groups
        ]
        self.final_lrs = [
            max_lr / (div_factor * final_div_factor)
            for _ in optimizer.param_groups
        ]

        for group, lr in zip(optimizer.param_groups, self.initial_lrs):
            group["lr"] = lr

        self.initial_momentums = [momentum_range[1] for _ in optimizer.param_groups]
        for group in optimizer.param_groups:
            if "betas" in group:
                group["betas"] = (momentum_range[1], group["betas"][1])
            elif "momentum" in group:
                group["momentum"] = momentum_range[1]

        super().__init__(optimizer, last_epoch)

    def _anneal(self, start, end, fraction):
        fraction = min(1.0, max(0.0, fraction))
        if self.anneal_strategy == "cos":
            factor = (1 - math.cos(math.pi * fraction)) / 2
        else:
            factor = fraction
        return start + (end - start) * factor

    def get_lr(self):
        step = min(self.last_epoch, self.total_steps - 1)
        up = self.step_size_up
        down_end = up + self.step_size_down
        annihilation_start = self.total_steps - self.annihilation_steps

        if step < up:
            fraction = step / max(1, up)
            return [
                self._anneal(base, self.max_lr, fraction)
                for base in self.initial_lrs
            ]

        if step < down_end:
            fraction = (step - up) / max(1, self.step_size_down)
            return [
                self._anneal(self.max_lr, base, fraction)
                for base in self.initial_lrs
            ]

        if self.annihilation_steps and step < self.total_steps:
            fraction = (step - annihilation_start) / self.annihilation_steps
            return [
                self._anneal(base, final, fraction)
                for base, final in zip(self.initial_lrs, self.final_lrs)
            ]

        if step >= self.total_steps - 1:
            return self.final_lrs

        return self.initial_lrs

    def get_momentum(self):
        step = min(self.last_epoch, self.total_steps - 1)
        fraction = min(1.0, step / max(1, self.step_size_up))
        momentum = (
            self.momentum_range[1]
            - fraction * (self.momentum_range[1] - self.momentum_range[0])
        )
        return [momentum for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        super().step(epoch)
        for group, momentum in zip(
            self.optimizer.param_groups, self.get_momentum()
        ):
            if "betas" in group:
                group["betas"] = (momentum, group["betas"][1])
            elif "momentum" in group:
                group["momentum"] = momentum


def get_scheduler(
    optimizer, ds_name="", name="ExponentialLR", step_size=1000,
    epochs=25, lr=None
):
    if lr is None:
        lr = optimizer.param_groups[0]["lr"]
    if lr <= 0:
        raise ValueError("lr must be positive.")

    if name == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    if name == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, step_size), gamma=0.1
        )

    if name == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 10, 15], gamma=0.1
        )

    if name == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )

    if name == "OneCycleLR":
        return CustomOneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=max(1, step_size),
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
            annihilation_frac=0.1,
        )

    if name == "CyclicLR":
        return optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr / 1000,
            max_lr=lr,
            step_size_up=max(1, step_size),
            mode="triangular2",
            cycle_momentum=False,
        )

    raise ValueError(f"Invalid schedule type: {name}")
