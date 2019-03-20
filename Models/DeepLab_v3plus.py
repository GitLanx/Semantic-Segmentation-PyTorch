import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV3Plus, self).__init__()

        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3])
        self.head = _DeepLabHead()
        self.decoder1 = nn.Conv2d(64, 48, 1)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(304, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        out, branch = self.resnet(x)
        _, _, uh, uw = branch.size()
        out = self.head(out)
        out = F.interpolate(out, size=(uh, uw), mode='bilinear', align_corners=True)
        branch = self.decoder1(branch)
        out = torch.cat([out, branch], 1)
        out = self.decoder2(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out

class _DeepLabHead(nn.Module):
    def __init__(self):
        super(_DeepLabHead, self).__init__()
        self.aspp = ASPP(2048, [6, 12, 18])     # output_stride = 16
        # self.aspp = ASPP(2048, [12, 24, 36])  # output_stride = 8
        # self.block = nn.Sequential(
            # nn.Conv2d(in_channels=256, out_channels=256,
            #           kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(num_features=256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
        #     nn.Conv2d(in_channels=256, out_channels=n_classes,
        #               kernel_size=1)
        # )

    def forward(self, x):
        out = self.aspp(x)
        # out = self.block(out)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        
        self.imagepool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.conv1 = self._ASPPConv(in_channels, out_channels, rate1)
        self.conv2 = self._ASPPConv(in_channels, out_channels, rate2)
        self.conv3 = self._ASPPConv(in_channels, out_channels, rate3)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels=5*out_channels, out_channels=out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        features1 = F.interpolate(self.imagepool(x), size=(h, w), mode='bilinear', align_corners=True)

        features2 = self.conv1x1(x)
        features3 = self.conv1(x)
        features4 = self.conv2(x)
        features5 = self.conv3(x)
        out = torch.cat((features1, features2, features3, features4, features5), 1)
        out = self.project(out)
        return out
    
    def _ASPPConv(self, in_channels, out_channels, atrous_rate):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=3, padding=atrous_rate,
                    dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        return block


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Adapted from https://github.com/speedinghzl/pytorch-segmentation-toolbox/blob/master/networks/deeplabv3.py
    """
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 2, 4))

        # for output_stride = 8
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 2, 4))

        self._initialize_weights()

    def _initialize_weights(self):
        resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1.load_state_dict(resnet.conv1.state_dict())
        self.bn1.load_state_dict(resnet.bn1.state_dict())
        self.layer1.load_state_dict(resnet.layer1.state_dict())
        self.layer2.load_state_dict(resnet.layer2.state_dict())
        self.layer3.load_state_dict(resnet.layer3.state_dict())
        self.layer4.load_state_dict(resnet.layer4.state_dict())

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        branch = self.maxpool(out)

        out = self.layer1(branch)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out, branch