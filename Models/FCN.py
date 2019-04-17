import numpy as np
import torch
import torch.nn as nn
import torchvision

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):
    """Adapted from official implementation:

    https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s/train.prototxt
    """
    def __init__(self, n_classes):
        super(FCN32s, self).__init__()

        features = []
        # conv1
        features.append(nn.Conv2d(3, 64, 3, padding=100))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(64, 64, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/2

        # conv2
        features.append(nn.Conv2d(64, 128, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(128, 128, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/4

        # conv3
        features.append(nn.Conv2d(128, 256, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(256, 256, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(256, 256, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/8

        # conv4
        features.append(nn.Conv2d(256, 512, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/16

        # conv5
        features.append(nn.Conv2d(512, 512, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.Conv2d(512, 512, 3, padding=1))
        features.append(nn.ReLU(inplace=True))
        features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/32

        self.features = nn.Sequential(*features)

        fc = []
        fc.append(nn.Conv2d(512, 4096, 7))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(p=0.5))
        fc.append(nn.Conv2d(4096, 4096, 1))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(p=0.5))
        self.fc = nn.Sequential(*fc)

        self.score_fr = nn.Conv2d(4096, n_classes, 1)
        self.upscore = nn.ConvTranspose2d(n_classes, n_classes, 64, stride=32,
                                          bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        self.score_fr.weight.data.zero_()
        self.score_fr.bias.data.zero_()

        assert self.upscore.kernel_size[0] == self.upscore.kernel_size[1]
        initial_weight = get_upsampling_weight(
                self.upscore.in_channels, self.upscore.out_channels,
                self.upscore.kernel_size[0])
        self.upscore.weight.data.copy_(initial_weight)


        vgg16 = torchvision.models.vgg16(pretrained=True)
        state_dict = vgg16.features.state_dict()
        self.features.load_state_dict(state_dict)

        for l1, l2 in zip(vgg16.classifier.children(), self.fc):
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Conv2d):
                l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
                l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

    def forward(self, x):
        out = self.features(x)
        out = self.fc(out)
        out = self.score_fr(out)
        out = self.upscore(out)
        out = out[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return out

    def get_parameters(self, bias=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight


class FCN8sAtOnce(nn.Module):
    def __init__(self, n_classes):
        super(FCN8sAtOnce, self).__init__()

        features1 = []
        # conv1
        features1.append(nn.Conv2d(3, 64, 3, padding=100))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/2

        # conv2
        features1.append(nn.Conv2d(64, 128, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(128, 128, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/4

        # conv3
        features1.append(nn.Conv2d(128, 256, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(256, 256, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(256, 256, 3, padding=1))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/8
        self.features1 = nn.Sequential(*features1)

        features2 = []
        # conv4
        features2.append(nn.Conv2d(256, 512, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(512, 512, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(512, 512, 3, padding=1))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/16
        self.features2 = nn.Sequential(*features2)

        features3 = []
        # conv5
        features3.append(nn.Conv2d(512, 512, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(512, 512, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(512, 512, 3, padding=1))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/32
        self.features3 = nn.Sequential(*features3)

        fc = []
        # fc6
        fc.append(nn.Conv2d(512, 4096, 7))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout2d())

        # fc7
        fc.append(nn.Conv2d(4096, 4096, 1))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout2d())
        self.fc = nn.Sequential(*fc)

        self.score_fr = nn.Conv2d(4096, n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)
        self.score_pool4 = nn.Conv2d(512, n_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_classes, n_classes, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in [self.score_fr, self.score_pool3, self.score_pool4]:
            m.weight.data.zero_()
            m.bias.data.zero_()

        for m in [self.upscore2, self.upscore8, self.upscore_pool4]:
            assert m.kernel_size[0] == m.kernel_size[1]
            initial_weight = get_upsampling_weight(
                m.in_channels, m.out_channels, m.kernel_size[0])
            m.weight.data.copy_(initial_weight)
        
        vgg16 = torchvision.models.vgg16(pretrained=True)
        vgg_features = [
            vgg16.features[:17],
            vgg16.features[17:24],
            vgg16.features[24:],
        ]
        features = [
            self.features1,
            self.features2,
            self.features3,
        ]

        for l1, l2 in zip(vgg_features, features):
            for ll1, ll2 in zip(l1.children(), l2.children()):
                if isinstance(ll1, nn.Conv2d) and isinstance(ll2, nn.Conv2d):
                    assert ll1.weight.size() == ll2.weight.size()
                    assert ll1.bias.size() == ll2.bias.size()
                    ll2.weight.data.copy_(ll1.weight.data)
                    ll2.bias.data.copy_(ll1.bias.data)

        for l1, l2 in zip(vgg16.classifier.children(), self.fc):
            if isinstance(l1, nn.Linear) and isinstance(l2, nn.Conv2d):
                l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
                l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

    def forward(self, x):
        pool3 = self.features1(x)       # 1/8
        pool4 = self.features2(pool3)   # 1/16
        pool5 = self.features3(pool4)     # 1/32
        fc = self.fc(pool5)
        score_fr = self.score_fr(fc)
        upscore2 = self.upscore2(score_fr)   # 1/16

        score_pool4 = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        score_pool4c = score_pool4[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        upscore_pool4 = self.upscore_pool4(upscore2 + score_pool4c)  # 1/8

        score_pool3 = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        score_pool3c = score_pool3[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        out = self.upscore8(upscore_pool4 + score_pool3c)

        out = out[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return out

    def get_parameters(self, bias=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight


if __name__ == "__main__":
    import torch
    import time
    model = FCN32s(21)
    print(f'==> Testing {model.__class__.__name__} with PyTorch')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.benchmark = True

    model = model.to(device)
    model.eval()

    x = torch.Tensor(1, 3, 500, 500)
    x = x.to(device)

    torch.cuda.synchronize()
    t_start = time.time()
    for i in range(10):
        model(x)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    print(f'Speed: {(elapsed_time / 10) * 1000:.2f} ms')