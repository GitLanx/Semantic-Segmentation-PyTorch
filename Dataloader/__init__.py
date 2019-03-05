from .loader import BaseLoader
from .voc_loader import VOCLoader, SBDLoader, VOC11Val
from .citys_loader import CityscapesLoader

VALID_DATASET = ['voc', 'cityscapes', 'sbd']

def get_loader(dataset_type):
    if dataset_type == '':
        return BaseLoader
    elif dataset_type == 'voc':
        return VOCLoader
    elif dataset_type == 'cityscapes':
        return CityscapesLoader
    elif dataset_type == 'sbd':
        return SBDLoader
    elif dataset_type == 'voc11':
        return VOC11Val
    else:
        raise ValueError('Unsupported dataset, '
                         'supported datasets as follows:\n{}\n'
                         'voc11 only for evaluation'.format(', '.join(VALID_DATASET)))