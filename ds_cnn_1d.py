import torch
import torch.nn as nn

class DSBlock1D_Weather(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        # Depthwise convolution
        self.dw = nn.Conv1d(in_ch, in_ch, 5, stride, 2, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = nn.SiLU()

        # Pointwise convolution
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.SiLU()

        # Residual shortcut
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.act1(self.bn1(self.dw(x)))
        x = self.bn2(self.pw(x))
        return self.act2(x + identity)


class DSCNN1D_Weather(nn.Module):
    """Depthwise-separable 1D CNN for univariate Weather forecasting.

    Input:  (B, in_channels, L) with in_channels = 1 target + n_calendar
            (default 3 = OT + tod_sin + tod_cos), L = lookback (e.g. 144).
    Params: 38,693 with in_channels=3, num_outputs=1.
    """

    def __init__(self, in_channels=3, num_outputs=1):
        super().__init__()

        # (B, 3, 144)
        self.conv1 = nn.Conv1d(in_channels, 24, 5, 1, 2, bias=False)
        self.bn1 = nn.BatchNorm1d(24)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)                      # 144 -> 72

        # Stage 1
        self.block1 = DSBlock1D_Weather(24, 24)           # 72
        self.block2 = DSBlock1D_Weather(24, 46, stride=2) # 72 -> 36

        # Stage 2
        self.block3 = DSBlock1D_Weather(46, 46)           # 36
        self.block4 = DSBlock1D_Weather(46, 93, stride=2) # 36 -> 18

        # Stage 3
        self.block5 = DSBlock1D_Weather(93, 93)           # 18
        self.block6 = DSBlock1D_Weather(93, 93)           # 18

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(93, 44)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(44, num_outputs)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.pool(x).squeeze(-1)

        x = self.act2(self.fc1(x))
        return self.fc2(x)


class DSBlock1D_Bike(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        # Depthwise convolution
        self.dw = nn.Conv1d(in_ch, in_ch, 5, stride, 2, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = nn.SiLU()

        # Pointwise convolution
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.SiLU()

        # Residual shortcut
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.act1(self.bn1(self.dw(x)))
        x = self.bn2(self.pw(x))
        return self.act2(x + identity)


class DSCNN1D_Bike(nn.Module):
    def __init__(self, in_channels=19, num_outputs=1):
        super().__init__()

        # Input: (B, 19, 24) — 10 historical + 9 calendar channels
        self.conv1 = nn.Conv1d(in_channels, 30, 5, 1, 2, bias=False)
        self.bn1 = nn.BatchNorm1d(30)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)                      # 24 -> 12

        # Stage 1
        self.block1 = DSBlock1D_Bike(30, 30)              # 12
        self.block2 = DSBlock1D_Bike(30, 58, stride=2)    # 12 -> 6

        # Stage 2
        self.block3 = DSBlock1D_Bike(58, 58)              # 6
        self.block4 = DSBlock1D_Bike(58, 119, stride=2)   # 6 -> 3

        # Stage 3
        self.block5 = DSBlock1D_Bike(119, 119)            # 3
        self.block6 = DSBlock1D_Bike(119, 119)            # 3

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(119, 68)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(68, num_outputs)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.pool(x).squeeze(-1)

        x = self.act2(self.fc1(x))
        return self.fc2(x)
        
class DSBlock1D_Crop(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.dw = nn.Conv1d(
            in_ch, in_ch, 5, stride, 2,
            groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.act1 = nn.SiLU()

        self.pw = nn.Conv1d(
            in_ch, out_ch, 1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.SiLU()

        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv1d(
                    in_ch, out_ch, 1, stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_ch)
            )
        )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.pw(x)
        x = self.bn2(x)

        x = x + identity
        x = self.act2(x)

        return x


class DSCNN1D_Crop(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        # Input: (B, 46)

        self.conv1 = nn.Conv1d(
            1, 16, 5, 2, 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(16)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        # Stage 1
        self.block1 = DSBlock1D_Crop(16, 16)

        self.block2 = DSBlock1D_Crop(
            16, 32, stride=2
        )

        # Stage 2
        self.block3 = DSBlock1D_Crop(32, 32)

        self.block4 = DSBlock1D_Crop(
            32, 63, stride=2
        )

        # Stage 3
        self.block5 = DSBlock1D_Crop(63, 63)

        self.block6 = DSBlock1D_Crop(63, 63)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(63, 24)
        self.act2 = nn.SiLU()

        self.fc2 = nn.Linear(
            24, num_classes
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.pool(x)
        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.act2(x)
        x = self.fc2(x)

        return x

