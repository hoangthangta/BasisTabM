import torch.nn as nn


class ResBlock1D_Weather(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.act(x + identity)


class ResNet1D_Weather(nn.Module):
    """1D ResNet for univariate Weather forecasting.

    Input:  (B, in_channels, L) with in_channels = 1 target + n_calendar
            (default 3 = OT + tod_sin + tod_cos), L = lookback (e.g. 144).
    Params: 38,693 with in_channels=3, num_outputs=1.
    """

    def __init__(self, in_channels=3, num_outputs=1):
        super().__init__()

        # (B, 3, 144)
        self.conv1 = nn.Conv1d(in_channels, 30, 5, 1, 2, bias=False)
        self.bn1 = nn.BatchNorm1d(30)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)                      # 144 -> 72

        self.block1 = ResBlock1D_Weather(30, 30)          # 72
        self.block2 = ResBlock1D_Weather(30, 52, stride=2)  # 72 -> 36
        self.block3 = ResBlock1D_Weather(52, 52)          # 36

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(52, 29)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(29, num_outputs)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x).squeeze(-1)

        x = self.act2(self.fc1(x))
        return self.fc2(x)


class ResBlock1D_Bike(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.act(x + identity)


class ResNet1D_Bike(nn.Module):
    def __init__(self, in_channels=19, num_outputs=1):
        super().__init__()

        # Input: (B, 19, 24) — 10 historical + 9 calendar channels
        self.conv1 = nn.Conv1d(in_channels, 28, 5, 1, 2, bias=False)
        self.bn1 = nn.BatchNorm1d(28)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)                    # 24 -> 12

        self.block1 = ResBlock1D_Bike(28, 28)           # 12
        self.block2 = ResBlock1D_Bike(28, 72, stride=2) # 12 -> 6
        self.block3 = ResBlock1D_Bike(72, 72)           # 6

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(72, 31)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(31, num_outputs)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x).squeeze(-1)

        x = self.act2(self.fc1(x))
        return self.fc2(x)
        
class ResBlock1D_Crop(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_ch, out_ch, 3, stride, 1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv1d(
            out_ch, out_ch, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_ch, out_ch, 1, stride, bias=False
                ),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return self.act(x + identity)


class ResNet1D_Crop(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        # Input: (B, 46) -> (B, 1, 46)
        self.conv1 = nn.Conv1d(
            1, 7, 5, 2, 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(7)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        self.block1 = ResBlock1D_Crop(7, 7)

        self.block2 = ResBlock1D_Crop(
            7, 38, stride=2
        )

        self.block3 = ResBlock1D_Crop(38, 38)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(38, 63)
        self.act2 = nn.SiLU()
        self.fc2 = nn.Linear(63, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.pool(x).squeeze(-1)

        x = self.act2(self.fc1(x))
        return self.fc2(x)