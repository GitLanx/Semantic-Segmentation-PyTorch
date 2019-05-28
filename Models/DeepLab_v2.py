import math
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabASPPVGG(nn.Module):
    """Adapted from official implementation:

    http://liangchiehchen.com/projects/DeepLabv2_vgg.html
    """
    def __init__(self, n_classes):
        super(DeepLabASPPVGG, self).__init__()

        features = []
        features.append(nn.Conv2d(3, 64, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(64, 64, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))

        features.append(nn.Conv2d(64, 128, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(128, 128, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))

        features.append(nn.Conv2d(128, 256, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(256, 256, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(256, 256, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))

        features.append(nn.Conv2d(256, 512, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True))

        features.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True))
        self.features = nn.Sequential(*features)

        # hole = 6
        fc1 = []
        fc1.append(nn.Conv2d(512, 1024, 3, padding=6, dilation=6))
        fc1.append(nn.ReLU(inplace=True))
        fc1.append(nn.Dropout(p=0.5))
        fc1.append(nn.Conv2d(1024, 1024, 1))
        fc1.append(nn.ReLU(inplace=True))
        fc1.append(nn.Dropout(p=0.5))
        self.fc1 = nn.Sequential(*fc1)
        self.fc1_score = nn.Conv2d(1024, n_classes, 1)

        # hole = 12
        fc2 = []
        fc2.append(nn.Conv2d(512, 1024, 3, padding=12, dilation=12))
        fc2.append(nn.ReLU(inplace=True))
        fc2.append(nn.Dropout(p=0.5))
        fc2.append(nn.Conv2d(1024, 1024, 1))
        fc2.append(nn.ReLU(inplace=True))
        fc2.append(nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(*fc2)
        self.fc2_score = nn.Conv2d(1024, n_classes, 1)

        # hole = 18
        fc3 = []
        fc3.append(nn.Conv2d(512, 1024, 3, padding=18, dilation=18))
        fc3.append(nn.ReLU(inplace=True))
        fc3.append(nn.Dropout(p=0.5))
        fc3.append(nn.Conv2d(1024, 1024, 1))
        fc3.append(nn.ReLU(inplace=True))
        fc3.append(nn.Dropout(p=0.5))
        self.fc3 = nn.Sequential(*fc3)
        self.fc3_score = nn.Conv2d(1024, n_classes, 1)

        # hole = 24
        fc4 = []
        fc4.append(nn.Conv2d(512, 1024, 3, padding=24, dilation=24))
        fc4.append(nn.ReLU(inplace=True))
        fc4.append(nn.Dropout(p=0.5))
        fc4.append(nn.Conv2d(1024, 1024, 1))
        fc4.append(nn.ReLU(inplace=True))
        fc4.append(nn.Dropout(p=0.5))
        self.fc4 = nn.Sequential(*fc4)
        self.fc4_score = nn.Conv2d(1024, n_classes, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.fc1_score, self.fc2_score, self.fc3_score, self.fc4_score]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

        # for module in [self.fc1, self.fc2, self.fc3, self.fc4]:
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d):
        #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #             nn.init.normal_(m.weight, std=math.sqrt(2. / n))
                    # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                    # nn.init.constant_(m.bias, 0)

        vgg = torchvision.models.vgg16(pretrained=True)
        state_dict = vgg.features.state_dict()
        self.features.load_state_dict(state_dict)

    def forward(self, x):
        _, _, h, w = x.size()
        out = self.features(x)
        fuse1 = self.fc1_score(self.fc1(out))
        fuse2 = self.fc2_score(self.fc2(out))
        fuse3 = self.fc3_score(self.fc3(out))
        fuse4 = self.fc4_score(self.fc4(out))
        out = fuse1 + fuse2 + fuse3 + fuse4
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

    def get_parameters(self, bias=False, score=False):
        if score:
            for m in [self.fc1_score, self.fc2_score, self.fc3_score, self.fc4_score]:
                if bias:
                    yield m.bias
                else:
                    yield m.weight
        else:
            for module in [self.features, self.fc1, self.fc2, self.fc3, self.fc4]:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        if bias:
                            yield m.bias
                        else:
                            yield m.weight

def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters():
            p.requires_grad = False


class DeepLabASPPResNet(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabASPPResNet, self).__init__()
        self.resnet = ResNet(Bottleneck, [3, 4, 23, 3])
        self.atrous_rates = [6, 12, 18, 24]
        self.aspp = ASPP(2048, self.atrous_rates, n_classes)     
        self.resnet.apply(freeze_bn)

    def forward(self, x):
        _, _, h, w = x.size()
        x2 = F.interpolate(x, size=(int(h * 0.75) + 1, int(w * 0.75) + 1), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x, size=(int(h * 0.5) + 1, int(w * 0.5) + 1), mode='bilinear', align_corners=True)
        x = self.aspp(self.resnet(x))
        x2 = self.aspp(self.resnet(x2))
        x3 = self.aspp(self.resnet(x3))

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)

        x3 = F.interpolate(x3, size=(h, w), mode='bilinear', align_corners=True)

        x4 = torch.max(torch.max(x, x2), x3)
        return x, x2, x3, x4

    def get_parameters(self, bias=False, score=False):
        if score:
            for m in self.aspp.modules():
                if isinstance(m, nn.Conv2d):
                    if bias:
                        yield m.bias
                    else:
                        yield m.weight
        else:
            for m in self.resnet.modules():
                for p in m.parameters():
                    if p.requires_grad:
                        yield p


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, n_classes):
        super(ASPP, self).__init__()

        rate1, rate2, rate3, rate4 = atrous_rates
        self.conv1 = nn.Conv2d(2048, n_classes, kernel_size=3, padding=rate1, dilation=rate1, bias=True)
        self.conv2 = nn.Conv2d(2048, n_classes, kernel_size=3, padding=rate2, dilation=rate2, bias=True)
        self.conv3 = nn.Conv2d(2048, n_classes, kernel_size=3, padding=rate3, dilation=rate3, bias=True)
        self.conv4 = nn.Conv2d(2048, n_classes, kernel_size=3, padding=rate4, dilation=rate4, bias=True)

        self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out')
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features1 = self.conv1(x)
        features2 = self.conv2(x)
        features3 = self.conv3(x)
        features4 = self.conv4(x)
        out = features1 + features2 + features3 + features4

        return out

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
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
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self._initialize_weights()

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out')
    #         elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

        resnet = torchvision.models.resnet101(pretrained=True)
        self.conv1.load_state_dict(resnet.conv1.state_dict())
        self.bn1.load_state_dict(resnet.bn1.state_dict())
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
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
