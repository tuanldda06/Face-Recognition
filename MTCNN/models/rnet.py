import torch
import torch.nn as nn
import torch.nn.Funtional as F

class RNet(nn.Module):
    def __init__(self) :
        super().__init__()

        self.conv1 = nn.Conv2d(3 , 28 , 3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3 , 2 , ceil_mode = True)


        self.conv2 = nn.Conv2d(28 , 48 , 3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3 , 2 , ceil_mode = True)

        self.conv3 = nn.Conv2d(48 , 64 , 2)
        self.prelu3 = nn.PReLU(64)

        self.dense4 = nn.Linear(576 , 128)
        self.prelu4 = nn.PReLU(128)

        #classification
        self.dense5_1 = nn.Linear(128,2)

        #Bouding box
        self.dense5_2 = nn.Linear(128 , 4)
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)

        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)

        x = self.prelu3(self.conv3(x))

        # Flatten theo đúng thứ tự pretrained model
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(x.shape[0], -1)

        x = self.prelu4(self.dense4(x))

        # Classification: Softmax
        cls = F.softmax(self.dense5_1(x), dim=1)

        # BBox Regression: Linear (Identity)
        bbox = self.dense5_2(x)

        return bbox, cls
