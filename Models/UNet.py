import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.enc1 = EncoderBlock(3, 64)
        self.enc1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = EncoderBlock(64, 128)
        self.enc2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = EncoderBlock(128, 256)
        self.enc3_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = EncoderBlock(256, 512)
        self.enc4_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = DecoderBlock(512, 1024, 512)
        self.dec4 = DecoderBlock(1024, 512, 256)
        self.dec3 = DecoderBlock(512, 256, 128)
        self.dec2 = DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1_pool = self.enc1_pool(enc1)
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.enc2_pool(enc2)
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.enc3_pool(enc3)
        enc4 = self.enc4(enc3_pool)
        enc4_pool = self.enc4_pool(enc4)
        center = self.center(enc4_pool)
        dec4 = self.dec4(torch.cat([center, enc4], 1))
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))
        final = self.final(dec1)
        return final


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
