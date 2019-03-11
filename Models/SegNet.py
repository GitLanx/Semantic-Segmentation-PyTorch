import torch.nn as nn
import torch.nn.functional as F


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
            nn.BatchNorm2d(in_channels),
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


class SegNet1(nn.Module):
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
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0.001)

    def forward(self, x):
        out = self.conv1(x)
        out, indices_1 = self.pool1(out)

        out = self.conv2(out)
        out, indices_2 = self.pool2(out)

        out = self.conv3(out)
        out, indices_3 = self.pool3(out)

        out = self.conv4(out)
        out, indices_4 = self.pool4(out)

        out = self.conv5(out)
        out, indices_5 = self.pool5(out)

        out = self.unpool6(out, indices_5)
        out = self.conv6_D(out)

        out = self.unpool7(out, indices_4)
        out = self.conv7_D(out)

        out = self.unpool8(out, indices_3)
        out = self.conv8_D(out)

        out = self.unpool9(out, indices_2)
        out = self.conv9_D(out)

        out = self.unpool10(out, indices_1)
        out = self.final(out)
        return out

    def copy_params_from_vgg16(self, vgg16):
        vgg_features = [
            vgg16.features[0:4],
            vgg16.features[5:9],
            vgg16.features[10:16],
            vgg16.features[17:23],
            vgg16.features[24:29]
        ]
        features = [
            self.conv1.encode,
            self.conv2.encode,
            self.conv3.encode,
            self.conv4.encode,
            self.conv5.encode
        ]
        for l1, l2 in zip(vgg_features, features):
            for i in range(len(list(l1.modules())) // 2):
                assert isinstance(l1[i * 2], nn.Conv2d) == isinstance(l2[i * 3], nn.Conv2d)
                assert l1[i * 2].weight.size() == l2[i * 3].weight.size()
                assert l1[i * 2].bias.size() == l2[i * 3].bias.size()
                l2[i * 3].weight.data = l1[i * 2].weight.data
                l2[i * 3].bias.data = l1[i * 2].bias.data


class SegNet(nn.Module):
    def __init__(self, n_classes):
        super(SegNet, self).__init__()
        # conv1
        self.conv1 = ConvBlock(3, 64, 2)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        # conv2
        self.conv2 = ConvBlock(64, 128, 2)
        # conv3
        self.conv3 = ConvBlock(128, 256, 3)
        # conv4
        self.conv4 = ConvBlock(256, 512, 3)
        # conv5
        self.conv5 = ConvBlock(512, 512, 3)

        self.conv6_D = DeconvBlock(512, 512, 3)

        self.conv7_D = DeconvBlock(512, 256, 3)

        self.conv8_D = DeconvBlock(256, 128, 3)

        self.conv9_D = DeconvBlock(128, 64, 2)

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0.001)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.pool(out)

        out = self.conv4(out)
        out = self.pool(out)

        out = self.conv5(out)
        out = self.pool(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.conv6_D(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.conv7_D(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.conv8_D(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.conv9_D(out)

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.final(out)
        return out

    def copy_params_from_vgg16(self, vgg16):
        vgg_features = [
            vgg16.features[0:4],
            vgg16.features[5:9],
            vgg16.features[10:16],
            vgg16.features[17:23],
            vgg16.features[24:29]
        ]
        features = [
            self.conv1.encode,
            self.conv2.encode,
            self.conv3.encode,
            self.conv4.encode,
            self.conv5.encode
        ]
        for l1, l2 in zip(vgg_features, features):
            for i in range(len(list(l1.modules())) // 2):
                assert isinstance(l1[i * 2], nn.Conv2d) == isinstance(l2[i * 3], nn.Conv2d)
                assert l1[i * 2].weight.size() == l2[i * 3].weight.size()
                assert l1[i * 2].bias.size() == l2[i * 3].bias.size()
                l2[i * 3].weight.data = l1[i * 2].weight.data
                l2[i * 3].bias.data = l1[i * 2].bias.data