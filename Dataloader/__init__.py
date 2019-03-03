from .loader import BaseLoader
from .voc_loader import VOCLoader, SBDLoader
from .citys_loader import CityscapesLoader


def get_loader(dataset_type):
    if dataset_type == '':
        return BaseLoader
    elif dataset_type == 'voc':
        return VOCLoader
    elif dataset_type == 'cityscapes':
        return CityscapesLoader
    elif dataset_type == 'sbd':
        return SBDLoader