import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabLargeFOV(nn.Module):
    """Adapted from official implementation:

    http://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-LargeFOV/train.prototxt

     input dimension equal to
     n = 32 * k - 31, e.g., 321 (for k = 11)
     Dimension after pooling w. subsampling:
     (16 * k - 15); (8 * k - 7); (4 * k - 3); (2 * k - 1); (k).
     For k = 11, these translate to  
               161;          81;          41;          21;  11
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
        features.append(nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True))
        self.features = nn.Sequential(*features)

        fc = []
        fc.append(nn.AvgPool2d(3, stride=1, padding=1))
        fc.append(nn.Conv2d(512, 1024, 3, padding=12, dilation=12))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Conv2d(1024, 1024, 1))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(p=0.5))
        self.fc = nn.Sequential(*fc)

        self.score = nn.Conv2d(1024, n_classes, 1)

        self._initialize_weights()

    def _initialize_weights(self):

        vgg = torchvision.models.vgg16(pretrained=True)
        state_dict = vgg.features.state_dict()
        self.features.load_state_dict(state_dict)

        # for m in self.fc.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.score.weight, std=0.01)
        nn.init.constant_(self.score.bias, 0)

    def forward(self, x):
        _, _, h, w = x.size()
        out = self.features(x)
        out = self.fc(out)
        out = self.score(out)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

    def get_parameters(self, bias=False, score=False):
        if score:
            if bias:
                yield self.score.bias
            else:
                yield self.score.weight
        else:
            for module in [self.features, self.fc]:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        if bias:
                            yield m.bias
                        else:
                            yield m.weight
            


class DeepLabMScLargeFOV(nn.Module):
    """Adapted from official implementation:

    http://www.cs.jhu.edu/~alanlab/ccvl/DeepLab-MSc-LargeFOV/train.prototxt

     input dimension equal to
     n = 32 * k - 31, e.g., 321 (for k = 11)
     Dimension after pooling w. subsampling:
     (16 * k - 15); (8 * k - 7); (4 * k - 3); (2 * k - 1); (k).
     For k = 11, these translate to  
               161;          81;          41;          21;  11
    """
    def __init__(self, n_classes):
        super(DeepLabMScLargeFOV, self).__init__()

        # from image to classifier
        self.MSc1 = self._msc(in_channels=3, stride=8)
        self.MSc1_score = nn.Conv2d(128, n_classes, 1)

        # Network
        features1 = []
        features1.append(nn.Conv2d(3, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))
        self.features1 = nn.Sequential(*features1)

        # first pool to classifier
        self.MSc2 = self._msc(in_channels=64, stride=4)
        self.MSc2_score = nn.Conv2d(128, n_classes, 1)

        # Network
        features2 = []
        features2.append(nn.Conv2d(64, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(128, 128, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True))
        self.features2 = nn.Sequential(*features2)

        # second pool to classifier
        self.MSc3 = self._msc(in_channels=128, stride=2)
        self.MSc3_score = nn.Conv2d(128, n_classes, 1)

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

        # third pool to classifier
        self.MSc4 = self._msc(in_channels=256, stride=1)
        self.MSc4_score = nn.Conv2d(128, n_classes, 1)

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

        # fourth pool to classifier
        self.MSc5 = self._msc(in_channels=512, stride=1)
        self.MSc5_score = nn.Conv2d(128, n_classes, 1)

        # Network
        features5 = []
        features5.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=2, dilation=2))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True))
        self.features5 = nn.Sequential(*features5)

        fc = []
        fc.append(nn.AvgPool2d(3, stride=1, padding=1))
        fc.append(nn.Conv2d(512, 1024, 3, padding=12, dilation=12))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout2d(p=0.5))
        fc.append(nn.Conv2d(1024, 1024, 1))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout2d(p=0.5))
        self.fc = nn.Sequential(*fc)

        self.score = nn.Conv2d(1024, n_classes, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # deeplablargefov_path = '/home/ecust/lx/Semantic-Segmentation-PyTorch/logs/deeplab-largefov_20190416_055924/model_best.pth.tar'
        # assert deeplablargefov_path is not None, ('If you are training DeepLab-MScLargeFov,'
        #     'Please specify the trained DeepLab-LargeFov\'s model path'
        #     'in DeepLab_v1.py/DeepLabMScLargeFOV/_initialize_weights')

        # checkpoint = torch.load(deeplablargefov_path)
        # trained_weights = checkpoint['model_state_dict']
        # trained_list = list(trained_weights.keys())

        # self_weights = self.features1.state_dict().copy()
        # self_list = list(self.features1.state_dict().keys())
        # for i in range(len(self_list)):
        #     self_weights[self_list[i]] = trained_weights[trained_list[:4][i]]
        # self.features1.load_state_dict(self_weights)

        # self_weights = self.features2.state_dict().copy()
        # self_list = list(self.features2.state_dict().keys())
        # for i in range(len(self_list)):
        #     self_weights[self_list[i]] = trained_weights[trained_list[4:8][i]]
        # self.features2.load_state_dict(self_weights)

        # self_weights = self.features3.state_dict().copy()
        # self_list = list(self.features3.state_dict().keys())
        # for i in range(len(self_list)):
        #     self_weights[self_list[i]] = trained_weights[trained_list[8:14][i]]
        # self.features3.load_state_dict(self_weights)

        # self_weights = self.features4.state_dict().copy()
        # self_list = list(self.features4.state_dict().keys())
        # for i in range(len(self_list)):
        #     self_weights[self_list[i]] = trained_weights[trained_list[14:20][i]]
        # self.features4.load_state_dict(self_weights)

        # self_weights = self.features5.state_dict().copy()
        # self_list = list(self.features5.state_dict().keys())
        # for i in range(len(self_list)):
        #     self_weights[self_list[i]] = trained_weights[trained_list[20:26][i]]
        # self.features5.load_state_dict(self_weights)

        # self_weights = self.fc.state_dict().copy()
        # self_list = list(self.fc.state_dict().keys())
        # for i in range(len(self_list)):
        #     self_weights[self_list[i]] = trained_weights[trained_list[26:30][i]]
        # self.fc.load_state_dict(self_weights)

        # self_weights = self.score.state_dict().copy()
        # self_list = list(self.score.state_dict().keys())
        # for i in range(len(self_list)):
        #     self_weights[self_list[i]] = trained_weights[trained_list[30:][i]]
        # self.score.load_state_dict(self_weights)

        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg_features = [
            vgg16.features[0:4],
            vgg16.features[5:9],
            vgg16.features[10:16],
            vgg16.features[17:23],
            vgg16.features[24:30]
        ]
        features = [
            self.features1,
            self.features2,
            self.features3,
            self.features4,
            self.features5
        ]
        for l1, l2 in zip(vgg_features, features):
            for ll1, ll2 in zip(l1.children(), l2.children()):
                if isinstance(ll1, nn.Conv2d) and isinstance(ll2, nn.Conv2d):
                    assert ll1.weight.size() == ll2.weight.size()
                    assert ll1.bias.size() == ll2.bias.size()
                    ll2.weight.data = ll1.weight.data
                    ll2.bias.data = ll1.bias.data

        for module in [self.MSc1, self.MSc2, self.MSc3, self.MSc4, self.MSc5]:
            for m in module.children():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

        for m in [self.MSc1_score, self.MSc2_score, self.MSc3_score,
                  self.MSc4_score, self.MSc5_score]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def _msc(self, in_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        fuse1 = self.MSc1(x)
        fuse1 = self.MSc1_score(fuse1)
        out = self.features1(x)
        fuse2 = self.MSc2(out)
        fuse2 = self.MSc2_score(fuse2)
        out = self.features2(out)
        fuse3 = self.MSc3(out)
        fuse3 = self.MSc3_score(fuse3)
        out = self.features3(out)
        fuse4 = self.MSc4(out)
        fuse4 = self.MSc4_score(fuse4)
        out = self.features4(out)
        fuse5 = self.MSc5(out)
        fuse5 = self.MSc5_score(fuse5)
        out = self.features5(out)
        out = self.fc(out)
        out = self.score(out)
        out = fuse1 + fuse2 + fuse3 + fuse4 + fuse5 + out
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        return out

    def get_parameters(self, bias=False, score=False):
        if score:
            if bias:
                yield self.score.bias
            else:
                yield self.score.weight
        else:
            for module in [self.MSc1, self.MSc2, self.MSc3, self.MSc4, self.MSc5,
                           self.MSc1_score, self.MSc2_score, self.MSc3_score, self.MSc4_score,
                           self.MSc5_score]:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        if bias:
                            yield m.bias
                        else:
                            yield m.weight

if __name__ == "__main__":
    import torch
    import time
    model = DeepLabMScLargeFOV(21)
    print(f'==> Testing {model.__class__.__name__} with PyTorch')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.benchmark = True

    model = model.to(device)
    model.eval()

    x = torch.Tensor(1, 3, 321, 321)
    x = x.to(device)

    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(10):
        model(x)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print(f'Speed: {(elapsed_time / 10) * 1000:.2f} ms')