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


if __name__ == "__main__":
    import torch
    import time
    model = DeepLabLargeFOV(21)
    print(f'==> Testing {model.__class__.__name__} with PyTorch')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
