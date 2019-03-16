import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabASPP(nn.Module):
    """
    official caffe training prototxt
    http://liangchiehchen.com/projects/DeepLabv2_vgg.html
    """
    def __init__(self, n_classes):
        super(DeepLabASPP, self).__init__()

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
        fc1.append(nn.Conv2d(1024, n_classes, 1))
        self.fc1 = nn.Sequential(*fc1)

        # hole = 12
        fc2 = []
        fc2.append(nn.Conv2d(512, 1024, 3, padding=12, dilation=12))
        fc2.append(nn.ReLU(inplace=True))
        fc2.append(nn.Dropout(p=0.5))
        fc2.append(nn.Conv2d(1024, 1024, 1))
        fc2.append(nn.ReLU(inplace=True))
        fc2.append(nn.Dropout(p=0.5))
        fc2.append(nn.Conv2d(1024, n_classes, 1))
        self.fc2 = nn.Sequential(*fc2)

        # hole = 18
        fc3 = []
        fc3.append(nn.Conv2d(512, 1024, 3, padding=18, dilation=18))
        fc3.append(nn.ReLU(inplace=True))
        fc3.append(nn.Dropout(p=0.5))
        fc3.append(nn.Conv2d(1024, 1024, 1))
        fc3.append(nn.ReLU(inplace=True))
        fc3.append(nn.Dropout(p=0.5))
        fc3.append(nn.Conv2d(1024, n_classes, 1))
        self.fc3 = nn.Sequential(*fc3)

        # hole = 24
        fc4 = []
        fc4.append(nn.Conv2d(512, 1024, 3, padding=24, dilation=24))
        fc4.append(nn.ReLU(inplace=True))
        fc4.append(nn.Dropout(p=0.5))
        fc4.append(nn.Conv2d(1024, 1024, 1))
        fc4.append(nn.ReLU(inplace=True))
        fc4.append(nn.Dropout(p=0.5))
        fc4.append(nn.Conv2d(1024, n_classes, 1))
        self.fc4 = nn.Sequential(*fc4)

        self._initialize_weights()

    def _initialize_weights(self):
        vgg = torchvision.models.vgg16(pretrained=True)
        state_dict = vgg.features.state_dict()
        self.features.load_state_dict(state_dict)

        # for m in self.classifier.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)

        # nn.init.normal_(self.score.weight, std=0.01)
        # nn.init.constant_(self.score.bias, 0)

    def forward(self, x):
        N, C, H, W = x.size()
        out = self.features(x)
        fuse1 = self.fc1(out)
        fuse2 = self.fc2(out)
        fuse3 = self.fc3(out)
        fuse4 = self.fc4(out)
        out = fuse1 + fuse2 + fuse3 + fuse4
        out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
        return out
