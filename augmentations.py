import random
import numpy as np
from PIL import Image, ImageOps


class Compose:
    def __init__(self, augmentations):
        self.augmentations = augmentations
    
    def __call__(self, imgs, lbls):
        assert imgs.size == lbls.size
        for aug in self.augmentations:
            imgs, lbls = aug(imgs, lbls)
        
        return imgs, lbls


class RandomFlip:
    def __init__(self, prob=0.5):
        super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return image, label


class RandomCrop:
    def __init__(self, crop_size):
        super(RandomCrop, self).__init__()
        self.crop_size = crop_size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, label):
        if image.size[0] < self.crop_size[1]:
            image = ImageOps.expand(image, (self.crop_size[1] - image.size[0], 0), fill=0)
            label = ImageOps.expand(label, (self.crop_size[1] - label.size[0], 0), fill=0)
        if image.size[1] < self.crop_size[0]:
            image = ImageOps.expand(image, (0, self.crop_size[0] - image.size[1]), fill=0)
            label = ImageOps.expand(label, (0, self.crop_size[0] - label.size[1]), fill=0)

        i, j, h, w = self.get_params(image, self.crop_size)
        image = image.crop((j, i, j + w, i + h))
        label = label.crop((j, i, j + w, i + h))

        return image, label


def get_augmentations(args):
    augs = []
    if args.flip:
        augs.append(RandomFlip())
    if args.crop_size:
        augs.append(RandomCrop(args.crop_size))

    if augs == []:
        return None
    print('Using augmentations: ', end=' ')
    for x in augs:
        print(x.__class__.__name__, end=' ')
    print('\n')

    return Compose(augs)