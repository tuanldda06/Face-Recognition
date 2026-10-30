import torch
import torch.nn as nn
import torch.nn.functional as F

class RNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (B, 3, 24, 24) -> Conv1: (B, 28, 22, 22) -> Pool1: (B, 28, 11, 11)
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Input: (B, 28, 11, 11) -> Conv2: (B, 48, 9, 9) -> Pool2: (B, 48, 4, 4)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Input: (B, 48, 4, 4) -> Conv3: (B, 64, 3, 3)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2, stride=1)
        self.prelu3 = nn.PReLU(64)

        # Flatten: 64 * 3 * 3 = 576 -> 128
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)

        # Nhánh Classification (2 lớp: Background / Face)
        self.dense5_1 = nn.Linear(128, 2)

        # Nhánh Bounding Box Regression (4 tọa độ offset)
        self.dense5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)

        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)

        x = self.prelu3(self.conv3(x))

        # Trải phẳng tensor từ (B, 64, 3, 3) thành (B, 576) để đưa vào lớp Linear
        x = torch.flatten(x, start_dim=1)

        x = self.prelu4(self.dense4(x))

        cls = self.dense5_1(x)
        bbox = self.dense5_2(x)
        return bbox, cls
