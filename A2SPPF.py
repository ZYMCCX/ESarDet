from torch import nn
import torch
import math
from CAANet import BaseConv, get_activation



class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.eca_1        = eca_block(hidden_channels)
        self.eca_2        = eca_block(hidden_channels)
        self.eca_3        = eca_block(hidden_channels)
        self.conv_1 = BaseConv(hidden_channels, hidden_channels, 1, stride=1, act=activation)
        self.conv_2 = dilateConv(hidden_channels, hidden_channels, 3, stride=1, act=activation ,dia=2)
        self.conv_3 = dilateConv(hidden_channels, hidden_channels, 3, stride=1, act=activation ,dia=3)
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):

        x = self.conv1(x)
        x1= self.conv_1(x)
        x1 = self.eca_1(x1)
        x2 = self.conv_2(x1)
        x2 =self.eca_2(x2)
        x3 = self.conv_3(x2)
        x3 = self.eca_3(x3)
        x = torch.cat((x ,x1 ,x2 ,x3), dim=1)
        x = self.conv2(x)
        return x


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class dilateConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu", dia=1):
        super().__init__()
        p = ksize+(ksize-1)*(dia-1)
        pad = (p - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias,dilation=dia)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))