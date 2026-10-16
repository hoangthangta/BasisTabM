import torch.nn as nn

class CNN1D_Weather(nn.Module):
    """1D CNN for univariate Weather forecasting.

    Input:  (B, in_channels, L) with in_channels = 1 target + n_calendar
            (default 3 = OT + tod_sin + tod_cos), L = lookback (e.g. 144).
    Params: 38,693 with in_channels=3, num_outputs=1.
    """

    def __init__(self, in_channels=3, num_outputs=1):
        super().__init__()

        # (B, 3, 144)
        self.conv1 = nn.Conv1d(in_channels, 35, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(35)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)          # 144 -> 72

        self.conv2 = nn.Conv1d(35, 84, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(84)
        self.act2 = nn.SiLU()
        self.pool2 = nn.MaxPool1d(2)          # 72 -> 36

        self.conv3 = nn.Conv1d(84, 51, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(51)
        self.act3 = nn.SiLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # 36 -> 1

        self.fc1 = nn.Linear(51, 29)
        self.act4 = nn.SiLU()
        self.fc2 = nn.Linear(29, num_outputs)

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = x.squeeze(-1)                     # (B, 51)
        return self.fc2(self.act4(self.fc1(x)))


class CNN1D_Bike(nn.Module):
    def __init__(self, in_channels=19, num_outputs=1):
        super().__init__()

        # Input: (B, 19, 24)
        self.conv1 = nn.Conv1d(in_channels, 41, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(41)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)          # 24 -> 12

        self.conv2 = nn.Conv1d(41, 105, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(105)
        self.act2 = nn.SiLU()
        self.pool2 = nn.MaxPool1d(2)          # 12 -> 6

        self.conv3 = nn.Conv1d(105, 70, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(70)
        self.act3 = nn.SiLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # 6 -> 1

        self.fc1 = nn.Linear(70, 34)
        self.act4 = nn.SiLU()
        self.fc2 = nn.Linear(34, num_outputs)

    def forward(self, x):
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = x.squeeze(-1)                     # (B, 70)
        return self.fc2(self.act4(self.fc1(x)))


class CNN1D_Crop(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        # Input: (B, 46) -> (B, 1, 46)
        self.conv1 = nn.Conv1d(1, 16, 80, 4, 38)
        self.bn1 = nn.BatchNorm1d(16)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 36, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(36)
        self.act2 = nn.SiLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(36, 54, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(54)
        self.act3 = nn.SiLU()
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(54, 58)
        self.act4 = nn.SiLU()
        self.fc2 = nn.Linear(58, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = x.squeeze(-1)

        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)

        return x