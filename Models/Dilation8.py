import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Dilation8(nn.Module):
    """Adapted from official dilated8 implementation:

    https://github.com/fyu/dilation/blob/master/models/dilation8_pascal_voc_deploy.prototxt
    """
    def __init__(self, n_classes):
        super(Dilation8, self).__init__()
        features1 = []
        # conv1
        features1.append(nn.Conv2d(3, 64, 3))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/2

        # conv2
        features1.append(nn.Conv2d(64, 128, 3))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(128, 128, 3))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/4

        # conv3
        features1.append(nn.Conv2d(128, 256, 3))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(256, 256, 3))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(256, 256, 3))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/8

        # conv4
        features1.append(nn.Conv2d(256, 512, 3))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(512, 512, 3))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(512, 512, 3))
        features1.append(nn.ReLU(inplace=True))
        self.features1 = nn.Sequential(*features1)

        # conv5
        features2 = []
        features2.append(nn.Conv2d(512, 512, 3, dilation=2))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(512, 512, 3, dilation=2))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(512, 512, 3, dilation=2))
        features2.append(nn.ReLU(inplace=True))
        self.features2 = nn.Sequential(*features2)

        fc = []
        fc.append(nn.Conv2d(512, 4096, 7, dilation=4))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(p=0.5))
        fc.append(nn.Conv2d(4096, 4096, 1))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(p=0.5))
        fc.append(nn.Conv2d(4096, n_classes, 1))
        self.fc = nn.Sequential(*fc)

        context = []
        context.append(nn.Conv2d(n_classes, 2 * n_classes, 3, padding=33))
        context.append(nn.ReLU(inplace=True))
        context.append(nn.Conv2d(2 * n_classes, 2 * n_classes, 3, padding=0))
        context.append(nn.ReLU(inplace=True))
        context.append(nn.Conv2d(2 * n_classes, 4 * n_classes, 3, dilation=2))
        context.append(nn.ReLU(inplace=True))
        context.append(nn.Conv2d(4 * n_classes, 8 * n_classes, 3, dilation=4))
        context.append(nn.ReLU(inplace=True))
        context.append(nn.Conv2d(8 * n_classes, 16 * n_classes, 3, dilation=8))
        context.append(nn.ReLU(inplace=True))
        context.append(nn.Conv2d(16 * n_classes, 32 * n_classes, 3, dilation=16))
        context.append(nn.ReLU(inplace=True))
        context.append(nn.Conv2d(32 * n_classes, 32 * n_classes, 3))
        context.append(nn.ReLU(inplace=True))
        context.append(nn.Conv2d(32 * n_classes, n_classes, 1))
        context.append(nn.ReLU(inplace=True))
        self.context = nn.Sequential(*context)

        self._initialize_weights()

    def _initialize_weights(self):
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg_features1 = vgg16.features[0:23]
        self.features1.load_state_dict(vgg_features1.state_dict())

        vgg_features2 = vgg16.features[24:30]
        for l1, l2 in zip(vgg_features2.children(), self.features2.children()):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

        fc = self.fc[0:4]
        for l1, l2 in zip(vgg16.classifier.children(), fc.children()):
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data.view(l2.weight.size())
                l2.bias.data = l1.bias.data.view(l2.bias.size())

        for m in self.context.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, h, w = x.size()
        out = self.features1(x)
        out = self.features2(out)
        out = self.fc(out)
        out = self.context(out)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out