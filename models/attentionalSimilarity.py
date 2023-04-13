# -*- coding: UTF-8 -*-
'''
@Project ：audioProcess 
@File    ：attentionalSimilarity.py
@Author  ：jiangym
@Date    ：2023/4/13 上午9:03
@Description :
'''


import torch.nn as nn
import torch.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(4, 4))
        self.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

net = ConvBlock(1, 32)
print(net)
