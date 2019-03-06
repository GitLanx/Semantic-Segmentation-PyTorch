import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ] * (n_blocks - 1)
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks):
        super(DeconvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ] * (n_blocks - 1)
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SegNet(nn.Module):
    def __init__(self, n_classes):
        super(SegNet, self).__init__()
        # conv1
        self.conv1 = ConvBlock(3, 64, 2)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/2

        # conv2
        self.conv2 = ConvBlock(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/4

        # conv3
        self.conv3 = ConvBlock(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/8

        # conv4
        self.conv4 = ConvBlock(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/16

        # conv5
        self.conv5 = ConvBlock(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/32

        self.unpool6 = nn.MaxUnpool2d(2, stride=2)
        self.conv6_D = DeconvBlock(512, 512, 3)

        self.unpool7 = nn.MaxUnpool2d(2, stride=2)
        self.conv7_D = DeconvBlock(512, 256, 3)

        self.unpool8 = nn.MaxUnpool2d(2, stride=2)
        self.conv8_D = DeconvBlock(256, 128, 3)

        self.unpool9 = nn.MaxUnpool2d(2, stride=2)
        self.conv9_D = DeconvBlock(128, 64, 2)

        self.unpool10 = nn.MaxUnpool2d(2, stride=2)
        self.final = nn.Sequential([
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
        ])

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h, indices_1 = self.pool1(h)

        h = self.conv2(h)
        h, indices_2 = self.pool2(h)

        h = self.conv3(h)
        h, indices_3 = self.pool3(h)

        h = self.conv4(h)
        h, indices_4 = self.pool4(h)

        h = self.conv5(h)
        h, indices_5 = self.pool5(h)

        h = self.unpool6(h, indices_5)
        h = self.conv6_D(h)

        h = self.unpool7(h, indices_4)
        h = self.conv7_D(h)

        h = self.unpool8(h, indices_3)
        h = self.conv8_D(h)

        h = self.unpool9(h, indices_2)
        h = self.conv9_D(h)

        h = self.unpool10(h, indices_1)
        h = self.final(h)
        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())