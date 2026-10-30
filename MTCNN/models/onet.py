import torch
import torch.nn as nn
import torch.nn.functional as F


class ONet(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)

        # Block 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # Block 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)

        # Fully Connected
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)

        # 3 output heads
        self.dense6_1 = nn.Linear(256, 2)   # Classification
        self.dense6_2 = nn.Linear(256, 4)   # BBox regression
        self.dense6_3 = nn.Linear(256, 10)  # 5 landmarks

    def forward(self, x):
        # CNN feature extraction
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)

        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)

        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)

        x = self.prelu4(self.conv4(x))

        # Flatten theo thứ tự của pretrained reference
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(x.shape[0], -1)

        # Fully Connected
        x = self.prelu5(self.dense5(x))

        # Classification: Softmax
        cls = F.softmax(self.dense6_1(x), dim=1)

        # Bounding Box Regression: Identity
        bbox = self.dense6_2(x)

        # Facial Landmarks: Identity
        landmarks = self.dense6_3(x)

        return bbox, landmarks, cls
