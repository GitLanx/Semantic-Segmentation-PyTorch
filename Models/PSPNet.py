import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F

# def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()

class PSPNet(nn.Module):
    """set crop size to 480
    """
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3])
        self.pyramid_pooling = PyramidPooling(2048, 512)
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(512, n_classes, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.final:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, h, w = x.size()
        out = self.resnet(x)
        out = self.pyramid_pooling(out)
        out = self.final(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        return out

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        self.pool1 = self._pyramid_conv(in_channels, out_channels, 10)
        self.pool2 = self._pyramid_conv(in_channels, out_channels, 20)
        self.pool3 = self._pyramid_conv(in_channels, out_channels, 30)
        self.pool4 = self._pyramid_conv(in_channels, out_channels, 60)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _pyramid_conv(self, in_channels, out_channels, scale):
        module = nn.Sequential(
            # nn.AdaptiveAvgPool2d(scale),
            nn.AvgPool2d(kernel_size=scale, stride=scale),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=.95),
            nn.ReLU(inplace=True)
        )
        return module

    def forward(self, x):
        _, _, h, w = x.size()
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)
        pool4 = self.pool4(x)
        pool1 = F.interpolate(pool1, size=(h, w), mode='bilinear', align_corners=True)
        pool2 = F.interpolate(pool2, size=(h, w), mode='bilinear', align_corners=True)
        pool3 = F.interpolate(pool3, size=(h, w), mode='bilinear', align_corners=True)
        pool4 = F.interpolate(pool4, size=(h, w), mode='bilinear', align_corners=True)
        out = torch.cat([x, pool1, pool2, pool3, pool4], 1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=.95)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=.95)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=.95)
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
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=.95),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.conv1.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for module in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        resnet = torchvision.models.resnet50(pretrained=True)
        self.layer1.load_state_dict(resnet.layer1.state_dict())
        self.layer2.load_state_dict(resnet.layer2.state_dict())
        self.layer3.load_state_dict(resnet.layer3.state_dict())
        self.layer4.load_state_dict(resnet.layer4.state_dict())

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.95))

        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out