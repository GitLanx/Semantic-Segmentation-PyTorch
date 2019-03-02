from .loader import BaseLoader
from .voc_loader import VOCLoader
from .citys_loader import CityscapesLoader


def get_loader(args):
    if args.dataset_type == '':
        return BaseLoader
    elif args.dataset_type == 'voc':
        return VOCLoader
    elif args.dataset_type == 'cityscapes':
        return CityscapesLoader
