# -*- coding: UTF-8 -*-
'''
@Project ：audioProcess 
@File    ：ACRNN.py
@Author  ：jiangym
@Date    ：2023/4/12 下午3:58
@Description :CNN+LSTM+Attention
'''

import torch
import torch.nn as nn
import torch.nn.functional as F



class ACRNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size):

        super(ACRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 4),
                               stride=(1, 1),
                               padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 3))

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(3, 1),
                               stride=(1, 1),
                               padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 1))

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(1, 5),
                               stride=(1, 1),
                               padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3))

        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()

        self.gru = nn.GRU(input_size=8960,
                          hidden_size=hidden_size,
                          num_layers=2,
                          dropout=0.5,
                          bidirectional=True)

        self.fc = nn.Linear(in_features=hidden_size*2, out_features=50)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool4(x)
        x = F.relu(x)

        x = self.flatten(x)

        # output 是所有隐藏层的状态，hidden是最后一层隐藏层的状态
        output, hidden = self.gru(x)
        output = self.fc(output)

        return output

if  __name__ == "__main__":
    network = ACRNN(in_channels=2, out_channels=32, hidden_size=256)
    input = torch.ones(32, 2, 128, 128)
    output = network(input)
    print(output.shape)
    print(network)






