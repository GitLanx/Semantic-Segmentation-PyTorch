from .FCN import FCN32s, FCN8sAtOnce
from .UNet import UNet
import torch
from torchvision import models

VALID_MODEL = ['fcn32s', 'fcn8s', 'unet']

def model_loader(model_name, n_classes, resume):
    if model_name == 'fcn32s':
        model = FCN32s(n_classes=n_classes)
    elif model_name == 'fcn8s':
        model = FCN8sAtOnce(n_classes=n_classes)
    elif model_name == 'unet':
        model = UNet(n_classes=n_classes)
    else:
        raise ValueError('Unsupported model, '
                         'supported models as follows:\n{}'.format(', '.join(VALID_MODEL)))

    start_epoch = 1
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        if model_name in ['fcn32s', 'fcn8s']:
            vgg16 = models.vgg16(pretrained=True)
            model.copy_params_from_vgg16(vgg16)
        checkpoint = None

    return model, start_epoch, checkpoint
