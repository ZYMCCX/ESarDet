import torch
from torch import nn
from CAANet import BaseConv,get_activation
from A2SPPF import eca_block


class ECCSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False,act="silu",convsize=1,dia=1):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # --------------------------------------------------#
        #   主干部分的初次卷积
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels,ksize=1, stride=1, act=act)

        # --------------------------------------------------#
        #   大的残差边部分的初次卷积
        # --------------------------------------------------#
        self.conv2 = D3Conv(in_channels, hidden_channels,convsize, stride=1, act=act,dia=dia)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        self.conv3 = BaseConv(2 * hidden_channels, out_channels,ksize=1, stride=1, act=act)

        self.eca = eca_block(in_channels)

        # --------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        # --------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in
                       range(n)]
        self.m = nn.Sequential(*module_list)
    def forward(self, x):
        # -------------------------------#
        #   x_1是主干部分
        # -------------------------------#
        x_1 = self.conv1(x)

        x_1 = self.eca(x_1)
        # -------------------------------#
        #   x_2是大的残差边部分
        # -------------------------------#
        x_2 = self.conv2(x)

        # -----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        # -----------------------------------------------#
        x_1 = self.m(x_1)
        # -----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        # -----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        return self.conv3(x)

class D3Conv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu",dia=1):
        super().__init__()

        p = ksize+(ksize-1)*(dia-1)
        pad = (p - 1) // 2

        self.dconv   = nn.Conv2d(in_channels, in_channels, kernel_size=ksize, stride=stride, padding=pad, groups=in_channels, dilation=dia,bias=False)
        self.dbn     = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.03)
        self.dact    = get_activation(act, inplace=True)

        self.pconv   = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.pbn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.pact    = get_activation(act, inplace=True)

    def forward(self, x):

        x = self.dact(self.dbn(self.dconv(x)))

        x = self.pact(self.pbn(self.pconv(x)))

        return x

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = BaseConv
        #--------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        #--------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        #--------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y