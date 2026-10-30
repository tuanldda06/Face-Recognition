import torch
import torch.nn as nn
import torch.nn.functional as F

class PNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (B, 3, 12, 12) -> Conv1: (B, 10, 10, 10) -> Pool1: (B, 10, 5, 5)
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # Input: (B, 10, 5, 5) -> Conv2: (B, 16, 3, 3)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(16)

        # Input: (B, 16, 3, 3) -> Conv3: (B, 32, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU(32)

        # Đầu ra phân loại mặt/không phải mặt (2 kênh) và hồi quy tọa độ (4 kênh)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)

        x = self.prelu2(self.conv2(x))
        # Lưu ý: P-Net không có pool2 ở đây

        x = self.prelu3(self.conv3(x))

        cls = F.softmax(self.conv4_1(x), dim=1)
        bbox = self.conv4_2(x)

        return bbox, cls