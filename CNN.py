import torch
import torch.nn as nn


class SimpleLeNet(nn.Module):

    def __init__(self):
        super(SimpleLeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),nn.Sigmoid(),#卷积层，输入通道1 输出通道6 卷积核5x5 填充2
            nn.AvgPool2d(kernel_size=2, stride=2),#平均池化层，核大小为2x2，步长为2
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),#输入通道数为6，输出通道数为16，卷积核大小为5x5。
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.net(x)
        return x