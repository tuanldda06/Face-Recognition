import torch
import torch.nn as nn
import torch.nn.functional as F


class ONet(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: (B, 3, 48, 48) -> Conv1: (B, 32, 46, 46) -> Pool1: (B, 32, 23, 23)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Block 2: (B, 32, 23, 23) -> Conv2: (B, 64, 21, 21) -> Pool2: (B, 64, 10, 10)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # Block 3: (B, 64, 10, 10) -> Conv3: (B, 64, 8, 8) -> Pool3: (B, 64, 4, 4)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Block 4: (B, 64, 4, 4) -> Conv4: (B, 128, 3, 3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.prelu4 = nn.PReLU(128)

        # Fully Connected: 128 * 3 * 3 = 1152 -> 256
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)

        # 3 nhánh đầu ra (Output heads)
        self.dense6_1 = nn.Linear(256, 2)   # Classification (Background / Face)
        self.dense6_2 = nn.Linear(256, 4)   # Bounding Box Regression (4 tọa độ)
        self.dense6_3 = nn.Linear(256, 10)  # Facial Landmarks (5 điểm x, y)

    def forward(self, x):
        # CNN feature extraction
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)

        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)

        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)

        x = self.prelu4(self.conv4(x))


        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(x.size(0), -1)

        x = self.prelu5(self.dense5(x))

        cls = F.softmax(self.dense6_1(x), dim=1)
        bbox = self.dense6_2(x)
        landmarks = self.dense6_3(x)

        return bbox, landmarks, cls