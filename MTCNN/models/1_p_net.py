import touch
import torch.nn as nn
import torch.nn.Functional as F

class PNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,10,3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPooling2d(2, 2, ceil_mode = True)

        self.conv2 = nn.Conv2d(3 , 16 , 3)
        self.prelu2 = nn.PReLU(16)

        self.conv3 = nn.Conv2d(3 , 32 , 3)
        self.prelu3 = nn.PReLU(32)


        self.conv4_1 = nn.Conv2d(32 , 2 , 1)
        self.conv4_2 = nn.Conv2d(32 , 4 , 1)
    def forward(self,x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)

        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)

        x = self.prelu3(self.conv3(x))

        cls = F.softmax(self.conv4_1(x) , dim= 1)
        bbox = self.conv_2(x)

        return cls , bbox
