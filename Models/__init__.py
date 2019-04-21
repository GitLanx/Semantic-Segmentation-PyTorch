from .FCN import FCN32s, FCN8sAtOnce
from .UNet import UNet
from .SegNet import SegNet
from .DeepLab_v1 import DeepLabLargeFOV
from .DeepLab_v2 import DeepLabASPPVGG, DeepLabASPPResNet
from .DeepLab_v3 import DeepLabV3
from .DeepLab_v3plus import DeepLabV3Plus
from .Dilation8 import Dilation8
from .PSPNet import PSPNet
import torch

VALID_MODEL = [
    'fcn32s', 'fcn8s', 'unet', 'segnet', 'deeplab-largefov', 'deeplab-aspp-vgg',
    'deeplab-aspp-resnet', 'deeplab-v3', 'deeplab-v3+', 'dilation8', 'pspnet'
]


def model_loader(model_name, n_classes, resume):
    model_name = model_name.lower()
    if model_name == 'fcn32s':
        model = FCN32s(n_classes=n_classes)
    elif model_name == 'fcn8s':
        model = FCN8sAtOnce(n_classes=n_classes)
    elif model_name == 'unet':
        model = UNet(n_classes=n_classes)
    elif model_name == 'segnet':
        model = SegNet(n_classes=n_classes)
    elif model_name == 'deeplab-largefov':
        model = DeepLabLargeFOV(n_classes=n_classes)
    elif model_name == 'deeplab-aspp-vgg':
        model = DeepLabASPPVGG(n_classes=n_classes)
    elif model_name == 'deeplab-aspp-resnet':
        model = DeepLabASPPResNet(n_classes=n_classes)
    elif model_name == 'deeplab-v3':
        model = DeepLabV3(n_classes=n_classes)
    elif model_name == 'deeplab-v3+':
        model = DeepLabV3Plus(n_classes=n_classes)
    elif model_name == 'dilation8':
        model = Dilation8(n_classes=n_classes)
    elif model_name == 'pspnet':
        model = PSPNet(n_classes=n_classes)
    else:
        raise ValueError('Unsupported model, '
                         'valid models as follows:\n{}'.format(
                             ', '.join(VALID_MODEL)))

    start_epoch = 1
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        checkpoint = None

    return model, start_epoch, checkpoint
