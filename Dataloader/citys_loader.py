import os
import torch
import numpy as np
from PIL import Image
from .baseloader import BaseLoader
from collections import namedtuple
from torch.utils import data
from torchvision import transforms


class CityscapesLoader(BaseLoader):
    """Cityscapes dataset loader.
    Parameters
    ----------
      root: path to cityscapes dataset.
        for directory:
        --VOCdevkit--VOC2012---ImageSets
                             |-JPEGImages
                             |-   ...
        root should be xxx/VOCdevkit/VOC2012
      n_classes: number of classes, default 19.
      split: choose subset of dataset, 'train','val' or 'trainval'.
      img_size: scale image to proper size.
      augmentations: whether to perform augmentation.
      ignore_index: ingore_index will be ignored in training phase and evaluation, default 255.
      class_weight: useful in unbalanced datasets.
      pretrained: whether to use pretrained models
    """

    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id',
                                                     'ignore_in_eval', 'color'])

    classes = [
        #               name                      id    trainId    ignoreInEval   color
        CityscapesClass('unlabeled',              0,    255,       True,          (0, 0, 0)),
        CityscapesClass('ego vehicle',            1,    255,       True,          (0, 0, 0)),
        CityscapesClass('rectification border',   2,    255,       True,          (0, 0, 0)),
        CityscapesClass('out of roi',             3,    255,       True,          (0, 0, 0)),
        CityscapesClass('static',                 4,    255,       True,          (0, 0, 0)),
        CityscapesClass('dynamic',                5,    255,       True,          (111, 74, 0)),
        CityscapesClass('ground',                 6,    255,       True,          (81, 0, 81)),
        CityscapesClass('road',                   7,    0,         False,         (128, 64, 128)),
        CityscapesClass('sidewalk',               8,    1,         False,         (244, 35, 232)),
        CityscapesClass('parking',                9,    255,       True,          (250, 170, 160)),
        CityscapesClass('rail track',             10,   255,       True,          (230, 150, 140)),
        CityscapesClass('building',               11,   2,         False,         (70, 70, 70)),
        CityscapesClass('wall',                   12,   3,         False,         (102, 102, 156)),
        CityscapesClass('fence',                  13,   4,         False,         (190, 153, 153)),
        CityscapesClass('guard rail',             14,   255,       True,          (180, 165, 180)),
        CityscapesClass('bridge',                 15,   255,       True,          (150, 100, 100)),
        CityscapesClass('tunnel',                 16,   255,       True,          (150, 120, 90)),
        CityscapesClass('pole',                   17,   5,         False,         (153, 153, 153)),
        CityscapesClass('polegroup',              18,   255,       True,          (153, 153, 153)),
        CityscapesClass('traffic light',          19,   6,         False,         (250, 170, 30)),
        CityscapesClass('traffic sign',           20,   7,         False,         (220, 220, 0)),
        CityscapesClass('vegetation',             21,   8,         False,         (107, 142, 35)),
        CityscapesClass('terrain',                22,   9,         False,         (152, 251, 152)),
        CityscapesClass('sky',                    23,   10,        False,         (70, 130, 180)),
        CityscapesClass('person',                 24,   11,        False,         (220, 20, 60)),
        CityscapesClass('rider',                  25,   12,        False,         (255, 0, 0)),
        CityscapesClass('car',                    26,   13,        False,         (0, 0, 142)),
        CityscapesClass('truck',                  27,   14,        False,         (0, 0, 70)),
        CityscapesClass('bus',                    28,   15,        False,         (0, 60, 100)),
        CityscapesClass('caravan',                29,   255,       True,          (0, 0, 90)),
        CityscapesClass('trailer',                30,   255,       True,          (0, 0, 110)),
        CityscapesClass('train',                  31,   16,        False,         (0, 80, 100)),
        CityscapesClass('motorcycle',             32,   17,        False,         (0, 0, 230)),
        CityscapesClass('bicycle',                33,   18,        False,         (119, 11, 32)),
        CityscapesClass('license plate',          -1,   -1,        True,          (0, 0, 142)),
    ]

    def __init__(
        self,
        root,
        n_classes=19,
        split="train",
        img_size=None,
        augmentations=None,
        ignore_index=255,
        class_weight=None,
        pretrained=False
        ):
        super(CityscapesLoader, self).__init__(root, n_classes, split, img_size, augmentations, ignore_index, class_weight, pretrained)

        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.labels_dir = os.path.join(self.root, 'gtFine', split)
        self.images = []
        self.labels = []

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26,
                              27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            label_dir = os.path.join(self.labels_dir, city)
            for file_name in os.listdir(img_dir):
                label_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                            'gtFine_labelIds.png')
                self.images.append(os.path.join(img_dir, file_name))
                self.labels.append(os.path.join(label_dir, label_name))

        print(f"Found {len(self.images)} {split} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        lbl = Image.open(self.labels[index])

        if self.img_size:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        if self.augmentations:
            img, lbl = self.augmentations(img, lbl)

        img, lbl = self.transform(img, lbl)
        return img, lbl

    def transform(self, img, lbl):
        img = self.tf(img)

        lbl = np.array(lbl, dtype=np.int32)
        lbl = self.encode_segmap(lbl)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def getpalette(self):
        return np.array([
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32]
        ])

    def decode_segmap(self, lbl):
        label_colours = self.getpalette()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to -1
        for _voidc in self.void_classes:
            mask[mask == _voidc] = -1
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    local_path = "/home/ecust/zww/DANet/datasets/cityscapes"
    dst = CityscapesLoader(local_path, transform=True)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples

        plt.subplots(1, 1)
        for j in range(1):
            plt.subplot(1, 2, j + 1)
            plt.imshow(np.transpose(imgs.numpy()[j], [1, 2, 0]))
            plt.subplot(1, 2, j + 2)
            plt.imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
