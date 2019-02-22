from FCN32s import FCN32s
import torch
from torchvision import models


def model_loader(model_name, n_classes, resume):
    if model_name == 'FCN32s':
        model = FCN32s(n_classes=n_classes)

    start_epoch = 1
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        if model_name == 'FCN32s':
            vgg16 = models.vgg16(pretrained=True)
            model.copy_params_from_vgg16(vgg16)

    return model, start_epoch, checkpoint
