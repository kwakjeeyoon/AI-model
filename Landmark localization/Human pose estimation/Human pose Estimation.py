import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

# Seed
torch.manual_seed(0)
# torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')

# 기본 convolutional block
class ResidualBlock(nn.Module):
    def __init__(self, num_channels=256):
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(num_channels, num_channels//2, kernel_size=1, bias=True)

        self.bn2 = nn.BatchNorm2d(num_channels//2)
        self.conv2 = nn.Conv2d(num_channels//2, num_channels//2, kernel_size=3, stride=1, padding=1, bias=True)

        self.bn3 = nn.BatchNorm2d(num_channels//2)
        self.conv3 = nn.Conv2d(num_channels//2, num_channels, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out

class Hourglass(nn.Module):
    def __init__(self, block, num_channels = 256):
        super(Hourglass, self).__init__()

        self.downconv_1 = block(num_channels)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.downconv_2 = block(num_channels)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.downconv_3 = block(num_channels)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        self.downconv_4 = block(num_channels)
        self.pool_4 = nn.MaxPool2d(kernel_size=2)

        self.midconv_1 = block(num_channels)
        self.midconv_2 = block(num_channels)
        self.midconv_3 = block(num_channels)

        self.skipconv_1 = block(num_channels)
        self.skipconv_2 = block(num_channels)
        self.skipconv_3 = block(num_channels)
        self.skipconv_4 = block(num_channels)

        self.upconv_1 = block(num_channels)
        self.upconv_2 = block(num_channels)
        self.upconv_3 = block(num_channels)
        self.upconv_4 = block(num_channels)

    def forward(self, x):
        x1 = self.downconv_1(x)
        x = self.pool_1(x1)
        x2 = self.downconv_2(x)
        x = self.pool_2(x2)
        x3 = self.downconv_3(x)
        x = self.pool_3(x3)
        x4 = self.downconv_4(x)
        x = self.pool_4(x4)

        x = self.midconv_1(x)
        x = self.midconv_2(x)
        x = self.midconv_3(x)

        x4 = self.skipconv_1(x4)
        x = F.upsample(x, scale_factor=2)
        x = x + x4
        x = self.upconv_1(x)

        x3 = self.skipconv_1(x3)
        x = F.upsample(x, scale_factor=2)
        x = x + x3
        x = self.upconv_2(x)

        x2 = self.skipconv_1(x2)
        x = F.upsample(x, scale_factor=2)
        x = x + x2
        x = self.upconv_3(x)

        x1 = self.skipconv_1(x1)
        x = F.upsample(x, scale_factor=2)
        x = x + x1
        x = self.upconv_4(x)

hg = Hourglass(ResidualBlock)

from torchsummary import summary
summary(hg, input_size = (256,64,64), device = 'cpu')

'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''

__all__ = ['HourglassNet','hg']

# Stacked Hourglass Network
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride = 1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out