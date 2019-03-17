import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV3, self).__init__()

        resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.head = _DeepLabHead(n_classes)

    
    def forward(self, x):
        _, _, h, w = x.size()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.head(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out

class _DeepLabHead(nn.Module):
    def __init__(self, n_classes):
        super(_DeepLabHead, self).__init__()
        self.aspp = ASPP(2048, [6, 12, 18])
        self.block = []
        self.block.append(nn.Conv2d(in_channels=256, out_channels=256,
                                    kernel_size=3, padding=1, bias=False))
        self.block.append(nn.BatchNorm2d(num_features=256))
        self.block.append(nn.ReLU(inplace=True))
        self.block.append(nn.Dropout(0.1))
        self.block.append(nn.Conv2d(in_channels=256, out_channels=n_classes,
                                    kernel_size=1))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        out = self.aspp(x)
        out = self.block(out)
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
            nn.Dropout(p=0.5)
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