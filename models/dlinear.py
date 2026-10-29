"""DLinear (Zeng et al., AAAI 2023, "Are Transformers Effective for Time Series Forecasting?").

The lookback window is split into a trend (moving average) and a seasonal (remainder)
component; each component is mapped to the forecast horizon by one linear layer over time,
and the two forecasts are summed.

Input : x of shape (B, L, C)   (L = lookback, C = channels)
Output: y of shape (B, H, C)   (H = horizon), one forecast per channel
"""
import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average along time with edge replication, so the output keeps length L."""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):                                   # x: (B, L, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)               # (B, L + 2*pad, C)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L, C)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend                             # seasonal, trend


class DLinear(nn.Module):
    """DLinear with optional per-channel ("individual") linear layers.

    Args:
        seq_len:     lookback length L
        pred_len:    forecast horizon H
        n_channels:  number of input channels C
        kernel_size: moving-average window for the trend (odd; 25 in the original paper)
        individual:  if True, each channel has its own pair of linear layers
    """

    def __init__(self, seq_len: int, pred_len: int, n_channels: int = 1,
                 kernel_size: int = 25, individual: bool = False):
        super().__init__()
        self.seq_len, self.pred_len = seq_len, pred_len
        self.n_channels, self.individual = n_channels, individual
        self.decomp = SeriesDecomp(kernel_size)

        def make_linear():
            lin = nn.Linear(seq_len, pred_len)
            # Original initialisation: every output starts as the mean of the window.
            nn.init.constant_(lin.weight, 1.0 / seq_len)
            nn.init.zeros_(lin.bias)
            return lin

        if individual:
            self.linear_seasonal = nn.ModuleList([make_linear() for _ in range(n_channels)])
            self.linear_trend = nn.ModuleList([make_linear() for _ in range(n_channels)])
        else:
            self.linear_seasonal = make_linear()
            self.linear_trend = make_linear()

    def forward(self, x):                                   # x: (B, L, C)
        if x.ndim != 3 or x.shape[1] != self.seq_len or x.shape[2] != self.n_channels:
            raise ValueError(f"DLinear expects input (B, {self.seq_len}, {self.n_channels}); "
                             f"got {tuple(x.shape)}")
        seasonal, trend = self.decomp(x)
        seasonal, trend = seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)   # (B, C, L)
        if self.individual:
            out = torch.stack([self.linear_seasonal[c](seasonal[:, c]) + self.linear_trend[c](trend[:, c])
                               for c in range(self.n_channels)], dim=1)       # (B, C, H)
        else:
            out = self.linear_seasonal(seasonal) + self.linear_trend(trend)   # (B, C, H)
        return out.permute(0, 2, 1)                                           # (B, H, C)
