from .custom_loader import CustomLoader
from .voc_loader import VOCLoader, SBDLoader, VOC11Val
from .citys_loader import CityscapesLoader
from .camvid_loader import CamVidLoader

VALID_DATASET = ['voc', 'cityscapes', 'sbd', 'voc11', 'camvid', 'custom']


def get_loader(dataset_type):
    if dataset_type.lower() == 'custom':
        return CustomLoader
    elif dataset_type.lower() == 'voc':
        return VOCLoader
    elif dataset_type.lower() == 'cityscapes':
        return CityscapesLoader
    elif dataset_type.lower() == 'sbd':
        return SBDLoader
    elif dataset_type.lower() == 'voc11':
        return VOC11Val
    elif dataset_type.lower() == 'camvid':
        return CamVidLoader
    else:
        raise ValueError('Unsupported dataset, '
                         'valid datasets as follows:\n{}\n'
                         'voc11 only for evaluation'.format(', '.join(VALID_DATASET)))
