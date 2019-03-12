import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabLargeFOV(nn.Module):
    """
    official caffe training prototxt
    http://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-LargeFOV/train.prototxt
    """
    def __init__(self, n_classes):
        super(DeepLabLargeFOV, self).__init__()

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
        features.append(nn.MaxPool2d(3, stride=1, padding=1))
        self.features = nn.Sequential(*features)

        classifier = []
        classifier.append(nn.AvgPool2d(3, stride=1, padding=1))
        classifier.append(nn.Conv2d(512, 1024, 3, padding=12, dilation=12))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Conv2d(1024, 1024, 1))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Dropout(p=0.5))
        self.classifier = nn.Sequential(*classifier)

        self.score = nn.Conv2d(1024, n_classes, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        vgg = torchvision.models.vgg16(pretrained=True)
        state_dict = vgg.features.state_dict()
        self.features.load_state_dict(state_dict)

        # for m in self.classifier.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.score.weight, std=0.01)
        nn.init.constant_(self.score.bias, 0)

    def forward(self, x):
        N, C, H, W = x.size()
        out = self.features(x)
        out = self.classifier(out)
        out = self.score(out)
        out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
        return out


class DeepLabMScLargeFOV(nn.Module):
    """
    official caffe training prototxt
    http://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-MSc-LargeFOV/train.prototxt
    """
    def __init__(self, n_classes):
        super(DeepLabMScLargeFOV, self).__init__()

        # from image to classifier
        MSc1 = []
        MSc1.append(nn.Conv2d(3, 128, 3, stride=8, padding=1))
        MSc1.append(nn.ReLU(inplace=True))
        MSc1.append(nn.Dropout(p=0.5))
        MSc1.append(nn.Conv2d(128, 128, 1))
        MSc1.append(nn.ReLU(inplace=True))
        MSc1.append(nn.Dropout(p=0.5))
        MSc1.append(nn.Conv2d(128, n_classes, 1))
        self.MSc1 = nn.Sequential(*MSc1)

        # Network
        features1 = []
        features1.append(nn.Conv2d(3, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))
        self.features1 = nn.Sequential(*features1)

        # first pool to classifier
        MSc2 = []
        MSc2.append(nn.Conv2d(64, 128, 3, stride=4, padding=1))
        MSc2.append(nn.ReLU(inplace=True))
        MSc2.append(nn.Dropout(p=0.5))
        MSc2.append(nn.Conv2d(128, 128, 1))
        MSc2.append(nn.ReLU(inplace=True))
        MSc2.append(nn.Dropout(p=0.5))
        MSc2.append(nn.Conv2d(128, n_classes, 1))
        self.MSc2 = nn.Sequential(*MSc2)

        # Network
        features2 = []
        features2.append(nn.Conv2d(64, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(128, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))
        self.features2 = nn.Sequential(*features2)

        #  second pool to classifier
        MSc3 = []
        MSc3.append(nn.Conv2d(128, 128, 3, stride=2, padding=1))
        MSc3.append(nn.ReLU(inplace=True))
        MSc3.append(nn.Dropout(p=0.5))
        MSc3.append(nn.Conv2d(128, 128, 1))
        MSc3.append(nn.ReLU(inplace=True))
        MSc3.append(nn.Dropout(p=0.5))
        MSc3.append(nn.Conv2d(128, n_classes, 1))
        self.MSc3 = nn.Sequential(*MSc3)

        # Network
        features3 = []
        features3.append(nn.Conv2d(128, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))
        self.features3 = nn.Sequential(*features3)

        #  third pool to classifier
        MSc4 = []
        MSc4.append(nn.Conv2d(256, 128, 3, stride=1, padding=1))
        MSc4.append(nn.ReLU(inplace=True))
        MSc4.append(nn.Dropout(p=0.5))
        MSc4.append(nn.Conv2d(128, 128, 1))
        MSc4.append(nn.ReLU(inplace=True))
        MSc4.append(nn.Dropout(p=0.5))
        MSc4.append(nn.Conv2d(128, n_classes, 1))
        self.MSc4 = nn.Sequential(*MSc4)

        # Network
        features4 = []
        features4.append(nn.Conv2d(256, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True))
        self.features4 = nn.Sequential(*features4)

        #  fourth pool to classifier
        MSc5 = []
        MSc5.append(nn.Conv2d(512, 128, 3, stride=1, padding=1))
        MSc5.append(nn.ReLU(inplace=True))
        MSc5.append(nn.Dropout(p=0.5))
        MSc5.append(nn.Conv2d(128, 128, 1))
        MSc5.append(nn.ReLU(inplace=True))
        MSc5.append(nn.Dropout(p=0.5))
        MSc5.append(nn.Conv2d(128, n_classes, 1))
        self.MSc5 = nn.Sequential(*MSc5)

        # Network
        features5 = []
        features5.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.MaxPool2d(3, stride=1, padding=1))
        self.features5 = nn.Sequential(*features5)

        classifier = []
        classifier.append(nn.AvgPool2d(3, stride=1, padding=1))
        classifier.append(nn.Conv2d(512, 1024, 3, padding=12, dilation=12))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Conv2d(1024, 1024, 1))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Conv2d(1024, n_classes, 1))
        self.classifier = nn.Sequential(*classifier)

        self._initialize_weights()

    def _initialize_weights(self):
        vgg = torchvision.models.vgg16(pretrained=True)
        state_dict = vgg.features.state_dict()
        self.features.load_state_dict(state_dict)

        # for m in self.MSc1.modules():
            
        # for m in self.classifier.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.score.weight, std=0.01)
        nn.init.constant_(self.score.bias, 0)

    def forward(self, x):
        N, C, H, W = x.size()
        out = self.features(x)
        out = self.classifier(out)
        out = self.score(out)
        out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
        return out