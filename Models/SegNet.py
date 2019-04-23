import torch.nn as nn
import torch.nn.functional as F
import torchvision


# use vgg16_bn pretrained model
class SegNet(nn.Module):
    """Adapted from official implementation:

    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/Models
    """
    def __init__(self, n_classes):
        super(SegNet, self).__init__()

        # conv1
        features1 = []
        features1.append(nn.Conv2d(3, 64, 3, padding=1))
        features1.append(nn.BatchNorm2d(64))
        features1.append(nn.ReLU(inplace=True))
        features1.append(nn.Conv2d(64, 64, 3, padding=1))
        features1.append(nn.BatchNorm2d(64))
        features1.append(nn.ReLU(inplace=True))
        self.features1 = nn.Sequential(*features1)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/2
        
        # conv2
        features2 = []
        features2.append(nn.Conv2d(64, 128, 3, padding=1))
        features2.append(nn.BatchNorm2d(128))
        features2.append(nn.ReLU(inplace=True))
        features2.append(nn.Conv2d(128, 128, 3, padding=1))
        features2.append(nn.BatchNorm2d(128))
        features2.append(nn.ReLU(inplace=True))
        self.features2 = nn.Sequential(*features2)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/4

        # conv3
        features3 = []
        features3.append(nn.Conv2d(128, 256, 3, padding=1))
        features3.append(nn.BatchNorm2d(256))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.BatchNorm2d(256))
        features3.append(nn.ReLU(inplace=True))
        features3.append(nn.Conv2d(256, 256, 3, padding=1))
        features3.append(nn.BatchNorm2d(256))
        features3.append(nn.ReLU(inplace=True))
        self.features3 = nn.Sequential(*features3)
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/8

        # conv4
        features4 = []
        features4.append(nn.Conv2d(256, 512, 3, padding=1))
        features4.append(nn.BatchNorm2d(512))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.BatchNorm2d(512))
        features4.append(nn.ReLU(inplace=True))
        features4.append(nn.Conv2d(512, 512, 3, padding=1))
        features4.append(nn.BatchNorm2d(512))
        features4.append(nn.ReLU(inplace=True))
        self.features4 = nn.Sequential(*features4)
        self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/16

        # conv5
        features5 = []
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.BatchNorm2d(512))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.BatchNorm2d(512))
        features5.append(nn.ReLU(inplace=True))
        features5.append(nn.Conv2d(512, 512, 3, padding=1))
        features5.append(nn.BatchNorm2d(512))
        features5.append(nn.ReLU(inplace=True))
        self.features5 = nn.Sequential(*features5)
        self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/32

        # convTranspose1
        self.unpool6 = nn.MaxUnpool2d(2, stride=2)
        features6 = []
        features6.append(nn.Conv2d(512, 512, 3, padding=1))
        features6.append(nn.BatchNorm2d(512))
        features6.append(nn.ReLU(inplace=True))
        features6.append(nn.Conv2d(512, 512, 3, padding=1))
        features6.append(nn.BatchNorm2d(512))
        features6.append(nn.ReLU(inplace=True))
        features6.append(nn.Conv2d(512, 512, 3, padding=1))
        features6.append(nn.BatchNorm2d(512))
        features6.append(nn.ReLU(inplace=True))
        self.features6 = nn.Sequential(*features6)

        # convTranspose2
        self.unpool7 = nn.MaxUnpool2d(2, stride=2)
        features7 = []
        features7.append(nn.Conv2d(512, 512, 3, padding=1))
        features7.append(nn.BatchNorm2d(512))
        features7.append(nn.ReLU(inplace=True))
        features7.append(nn.Conv2d(512, 512, 3, padding=1))
        features7.append(nn.BatchNorm2d(512))
        features7.append(nn.ReLU(inplace=True))
        features7.append(nn.Conv2d(512, 256, 3, padding=1))
        features7.append(nn.BatchNorm2d(256))
        features7.append(nn.ReLU(inplace=True))
        self.features7 = nn.Sequential(*features7)

        # convTranspose3
        self.unpool8 = nn.MaxUnpool2d(2, stride=2)
        features8 = []
        features8.append(nn.Conv2d(256, 256, 3, padding=1))
        features8.append(nn.BatchNorm2d(256))
        features8.append(nn.ReLU(inplace=True))
        features8.append(nn.Conv2d(256, 256, 3, padding=1))
        features8.append(nn.BatchNorm2d(256))
        features8.append(nn.ReLU(inplace=True))
        features8.append(nn.Conv2d(256, 128, 3, padding=1))
        features8.append(nn.BatchNorm2d(128))
        features8.append(nn.ReLU(inplace=True))
        self.features8 = nn.Sequential(*features8)

        # convTranspose4
        self.unpool9 = nn.MaxUnpool2d(2, stride=2)
        features9 = []
        features9.append(nn.Conv2d(128, 128, 3, padding=1))
        features9.append(nn.BatchNorm2d(128))
        features9.append(nn.ReLU(inplace=True))
        features9.append(nn.Conv2d(128, 64, 3, padding=1))
        features9.append(nn.BatchNorm2d(64))
        features9.append(nn.ReLU(inplace=True))
        self.features9 = nn.Sequential(*features9)

        # convTranspose5
        self.unpool10 = nn.MaxUnpool2d(2, stride=2)
        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            # if isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0.001)

        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        vgg_features = [
            vgg16.features[0:6],
            vgg16.features[7:13],
            vgg16.features[14:23],
            vgg16.features[24:33],
            vgg16.features[34:43]
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
                if isinstance(ll1, nn.BatchNorm2d) and isinstance(ll2, nn.BatchNorm2d):
                    assert ll1.weight.size() == ll2.weight.size()
                    assert ll1.bias.size() == ll2.bias.size()
                    ll2.weight.data = ll1.weight.data
                    ll2.bias.data = ll1.bias.data

    def forward(self, x):
        out = self.features1(x)
        out, indices_1 = self.pool1(out)
        out = self.features2(out)
        out, indices_2 = self.pool2(out)
        out = self.features3(out)
        out, indices_3 = self.pool3(out)
        out = self.features4(out)
        out, indices_4 = self.pool4(out)
        out = self.features5(out)
        out, indices_5 = self.pool5(out)
        out = self.unpool6(out, indices_5)
        out = self.features6(out)
        out = self.unpool7(out, indices_4)
        out = self.features7(out)
        out = self.unpool8(out, indices_3)
        out = self.features8(out)
        out = self.unpool9(out, indices_2)
        out = self.features9(out)
        out = self.unpool10(out, indices_1)
        out = self.final(out)
        return out


# use vgg16 pretrained model
# class SegNet(nn.Module):
#     """
#     Adapted from official implementation:

#     https://github.com/alexgkendall/SegNet-Tutorial/tree/master/Models
#     """
#     def __init__(self, n_classes):
#         super(SegNet, self).__init__()

#         # conv1
#         features1 = []
#         features1.append(nn.Conv2d(3, 64, 3, padding=1))
#         features1.append(nn.BatchNorm2d(64))
#         features1.append(nn.ReLU(inplace=True))
#         features1.append(nn.Conv2d(64, 64, 3, padding=1))
#         features1.append(nn.BatchNorm2d(64))
#         features1.append(nn.ReLU(inplace=True))
#         self.features1 = nn.Sequential(*features1)
#         self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/2
        
#         # conv2
#         features2 = []
#         features2.append(nn.Conv2d(64, 128, 3, padding=1))
#         features2.append(nn.BatchNorm2d(128))
#         features2.append(nn.ReLU(inplace=True))
#         features2.append(nn.Conv2d(128, 128, 3, padding=1))
#         features2.append(nn.BatchNorm2d(128))
#         features2.append(nn.ReLU(inplace=True))
#         self.features2 = nn.Sequential(*features2)
#         self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/4

#         # conv3
#         features3 = []
#         features3.append(nn.Conv2d(128, 256, 3, padding=1))
#         features3.append(nn.BatchNorm2d(256))
#         features3.append(nn.ReLU(inplace=True))
#         features3.append(nn.Conv2d(256, 256, 3, padding=1))
#         features3.append(nn.BatchNorm2d(256))
#         features3.append(nn.ReLU(inplace=True))
#         features3.append(nn.Conv2d(256, 256, 3, padding=1))
#         features3.append(nn.BatchNorm2d(256))
#         features3.append(nn.ReLU(inplace=True))
#         self.features3 = nn.Sequential(*features3)
#         self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/8

#         # conv4
#         features4 = []
#         features4.append(nn.Conv2d(256, 512, 3, padding=1))
#         features4.append(nn.BatchNorm2d(512))
#         features4.append(nn.ReLU(inplace=True))
#         features4.append(nn.Conv2d(512, 512, 3, padding=1))
#         features4.append(nn.BatchNorm2d(512))
#         features4.append(nn.ReLU(inplace=True))
#         features4.append(nn.Conv2d(512, 512, 3, padding=1))
#         features4.append(nn.BatchNorm2d(512))
#         features4.append(nn.ReLU(inplace=True))
#         self.features4 = nn.Sequential(*features4)
#         self.pool4 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/16

#         # conv5
#         features5 = []
#         features5.append(nn.Conv2d(512, 512, 3, padding=1))
#         features5.append(nn.BatchNorm2d(512))
#         features5.append(nn.ReLU(inplace=True))
#         features5.append(nn.Conv2d(512, 512, 3, padding=1))
#         features5.append(nn.BatchNorm2d(512))
#         features5.append(nn.ReLU(inplace=True))
#         features5.append(nn.Conv2d(512, 512, 3, padding=1))
#         features5.append(nn.BatchNorm2d(512))
#         features5.append(nn.ReLU(inplace=True))
#         self.features5 = nn.Sequential(*features5)
#         self.pool5 = nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True)  # 1/32

#         # convTranspose1
#         self.unpool6 = nn.MaxUnpool2d(2, stride=2)
#         features6 = []
#         features6.append(nn.Conv2d(512, 512, 3, padding=1))
#         features6.append(nn.BatchNorm2d(512))
#         features6.append(nn.ReLU(inplace=True))
#         features6.append(nn.Conv2d(512, 512, 3, padding=1))
#         features6.append(nn.BatchNorm2d(512))
#         features6.append(nn.ReLU(inplace=True))
#         features6.append(nn.Conv2d(512, 512, 3, padding=1))
#         features6.append(nn.BatchNorm2d(512))
#         features6.append(nn.ReLU(inplace=True))
#         self.features6 = nn.Sequential(*features6)

#         # convTranspose2
#         self.unpool7 = nn.MaxUnpool2d(2, stride=2)
#         features7 = []
#         features7.append(nn.Conv2d(512, 512, 3, padding=1))
#         features7.append(nn.BatchNorm2d(512))
#         features7.append(nn.ReLU(inplace=True))
#         features7.append(nn.Conv2d(512, 512, 3, padding=1))
#         features7.append(nn.BatchNorm2d(512))
#         features7.append(nn.ReLU(inplace=True))
#         features7.append(nn.Conv2d(512, 256, 3, padding=1))
#         features7.append(nn.BatchNorm2d(256))
#         features7.append(nn.ReLU(inplace=True))
#         self.features7 = nn.Sequential(*features7)

#         # convTranspose3
#         self.unpool8 = nn.MaxUnpool2d(2, stride=2)
#         features8 = []
#         features8.append(nn.Conv2d(256, 256, 3, padding=1))
#         features8.append(nn.BatchNorm2d(256))
#         features8.append(nn.ReLU(inplace=True))
#         features8.append(nn.Conv2d(256, 256, 3, padding=1))
#         features8.append(nn.BatchNorm2d(256))
#         features8.append(nn.ReLU(inplace=True))
#         features8.append(nn.Conv2d(256, 128, 3, padding=1))
#         features8.append(nn.BatchNorm2d(128))
#         features8.append(nn.ReLU(inplace=True))
#         self.features8 = nn.Sequential(*features8)

#         # convTranspose4
#         self.unpool9 = nn.MaxUnpool2d(2, stride=2)
#         features9 = []
#         features9.append(nn.Conv2d(128, 128, 3, padding=1))
#         features9.append(nn.BatchNorm2d(128))
#         features9.append(nn.ReLU(inplace=True))
#         features9.append(nn.Conv2d(128, 64, 3, padding=1))
#         features9.append(nn.BatchNorm2d(64))
#         features9.append(nn.ReLU(inplace=True))
#         self.features9 = nn.Sequential(*features9)

#         # convTranspose5
#         self.unpool10 = nn.MaxUnpool2d(2, stride=2)
#         self.final = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)

#         vgg16 = torchvision.models.vgg16(pretrained=True)
#         vgg_features = [
#             vgg16.features[0:4],
#             vgg16.features[5:9],
#             vgg16.features[10:16],
#             vgg16.features[17:23],
#             vgg16.features[24:29]
#         ]
#         features = [
#             self.features1,
#             self.features2,
#             self.features3,
#             self.features4,
#             self.features5
#         ]
#         for l1, l2 in zip(vgg_features, features):
#             for i in range(len(list(l1.modules())) // 2):
#                 assert isinstance(l1[i * 2], nn.Conv2d) == isinstance(l2[i * 3], nn.Conv2d)
#                 assert l1[i * 2].weight.size() == l2[i * 3].weight.size()
#                 assert l1[i * 2].bias.size() == l2[i * 3].bias.size()
#                 l2[i * 3].weight.data = l1[i * 2].weight.data
#                 l2[i * 3].bias.data = l1[i * 2].bias.data

#     def forward(self, x):
#         out = self.features1(x)
#         out, indices_1 = self.pool1(out)
#         out = self.features2(out)
#         out, indices_2 = self.pool2(out)
#         out = self.features3(out)
#         out, indices_3 = self.pool3(out)
#         out = self.features4(out)
#         out, indices_4 = self.pool4(out)
#         out = self.features5(out)
#         out, indices_5 = self.pool5(out)
#         out = self.unpool6(out, indices_5)
#         out = self.features6(out)
#         out = self.unpool7(out, indices_4)
#         out = self.features7(out)
#         out = self.unpool8(out, indices_3)
#         out = self.features8(out)
#         out = self.unpool9(out, indices_2)
#         out = self.features9(out)
#         out = self.unpool10(out, indices_1)
#         out = self.final(out)
#         return out


# Bilinear interpolation upsampling version
# class SegNet(nn.Module):
#     def __init__(self, n_classes):
#         super(SegNet, self).__init__()
#
#         # conv1
#         features = []
#         features.append(nn.Conv2d(3, 64, 3, padding=1))
#         features.append(nn.BatchNorm2d(64))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.Conv2d(64, 64, 3, padding=1))
#         features.append(nn.BatchNorm2d(64))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/2
        
#         # conv2
#         features.append(nn.Conv2d(64, 128, 3, padding=1))
#         features.append(nn.BatchNorm2d(128))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.Conv2d(128, 128, 3, padding=1))
#         features.append(nn.BatchNorm2d(128))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/4

#         # conv3
#         features.append(nn.Conv2d(128, 256, 3, padding=1))
#         features.append(nn.BatchNorm2d(256))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.Conv2d(256, 256, 3, padding=1))
#         features.append(nn.BatchNorm2d(256))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.Conv2d(256, 256, 3, padding=1))
#         features.append(nn.BatchNorm2d(256))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/8

#         # conv4
#         features.append(nn.Conv2d(256, 512, 3, padding=1))
#         features.append(nn.BatchNorm2d(512))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.Conv2d(512, 512, 3, padding=1))
#         features.append(nn.BatchNorm2d(512))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.Conv2d(512, 512, 3, padding=1))
#         features.append(nn.BatchNorm2d(512))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/16

#         # conv5
#         features.append(nn.Conv2d(512, 512, 3, padding=1))
#         features.append(nn.BatchNorm2d(512))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.Conv2d(512, 512, 3, padding=1))
#         features.append(nn.BatchNorm2d(512))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.Conv2d(512, 512, 3, padding=1))
#         features.append(nn.BatchNorm2d(512))
#         features.append(nn.ReLU(inplace=True))
#         features.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))  # 1/32
#         self.features = nn.Sequential(*features)

#         # convTranspose1
#         up1 = []
#         up1.append(nn.Conv2d(512, 512, 3, padding=1))
#         up1.append(nn.BatchNorm2d(512))
#         up1.append(nn.ReLU(inplace=True))
#         up1.append(nn.Conv2d(512, 512, 3, padding=1))
#         up1.append(nn.BatchNorm2d(512))
#         up1.append(nn.ReLU(inplace=True))
#         up1.append(nn.Conv2d(512, 512, 3, padding=1))
#         up1.append(nn.BatchNorm2d(512))
#         up1.append(nn.ReLU(inplace=True))
#         self.up1 = nn.Sequential(*up1)

#         # convTranspose2
#         up2 = []
#         up2.append(nn.Conv2d(512, 512, 3, padding=1))
#         up2.append(nn.BatchNorm2d(512))
#         up2.append(nn.ReLU(inplace=True))
#         up2.append(nn.Conv2d(512, 512, 3, padding=1))
#         up2.append(nn.BatchNorm2d(512))
#         up2.append(nn.ReLU(inplace=True))
#         up2.append(nn.Conv2d(512, 256, 3, padding=1))
#         up2.append(nn.BatchNorm2d(256))
#         up2.append(nn.ReLU(inplace=True))
#         self.up2 = nn.Sequential(*up2)

#         # convTranspose3
#         up3 = []
#         up3.append(nn.Conv2d(256, 256, 3, padding=1))
#         up3.append(nn.BatchNorm2d(256))
#         up3.append(nn.ReLU(inplace=True))
#         up3.append(nn.Conv2d(256, 256, 3, padding=1))
#         up3.append(nn.BatchNorm2d(256))
#         up3.append(nn.ReLU(inplace=True))
#         up3.append(nn.Conv2d(256, 128, 3, padding=1))
#         up3.append(nn.BatchNorm2d(128))
#         up3.append(nn.ReLU(inplace=True))
#         self.up3 = nn.Sequential(*up3)

#         # convTranspose4
#         up4 = []
#         up4.append(nn.Conv2d(128, 128, 3, padding=1))
#         up4.append(nn.BatchNorm2d(128))
#         up4.append(nn.ReLU(inplace=True))
#         up4.append(nn.Conv2d(128, 64, 3, padding=1))
#         up4.append(nn.BatchNorm2d(64))
#         up4.append(nn.ReLU(inplace=True))
#         self.up4 = nn.Sequential(*up4)

#         self.final = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, n_classes, kernel_size=3, padding=1),
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0.001)

#         vgg16 = torchvision.models.vgg16_bn(pretrained=True)
#         state_dict = vgg16.features.state_dict()
#         self.features.load_state_dict(state_dict)

#     def forward(self, x):
#         out = self.features(x)
#         out = F.interpolate(out, scale_factor=2, mode='bilinear')
#         out = self.up1(out)
#         out = F.interpolate(out, scale_factor=2, mode='bilinear')
#         out = self.up2(out)
#         out = F.interpolate(out, scale_factor=2, mode='bilinear')
#         out = self.up3(out)
#         out = F.interpolate(out, scale_factor=2, mode='bilinear')
#         out = self.up4(out)
#         out = F.interpolate(out, scale_factor=2, mode='bilinear')
#         out = self.final(out)
#         return out
