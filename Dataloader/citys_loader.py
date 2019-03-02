import os
import torch
import numpy as np
from PIL import Image
from collections import namedtuple
from torch.utils import data
from torchvision import transforms


class CityscapesLoader(data.Dataset):
    """cityscapesLoader
    Adapted from:
    https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py
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
        split="train",
        transform=False,
        img_size=None,
        augmentations=None
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = transform
        self.augmentations = augmentations
        self.n_classes = 19
        self.img_size = img_size
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.labels_dir = os.path.join(self.root, 'gtFine', split)
        self.images = []
        self.labels = []
        self.tf = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.untf = transforms.Compose(
            [transforms.ToPILImage()]
        )
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26,
                              27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))

        self.cityspallete = [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]

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
        """__len__"""
        return len(self.images)

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img = Image.open(self.images[index]).convert('RGB')
        lbl = Image.open(self.labels[index])

        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = self.tf(img)
        lbl = np.array(lbl, dtype=np.int32)
        lbl = self.encode_segmap(lbl)

        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, lbl):
        # r = temp.copy()
        # g = temp.copy()
        # b = temp.copy()
        # for l in range(0, self.n_classes):
        #     r[temp == l] = self.label_colours[l][0]
        #     g[temp == l] = self.label_colours[l][1]
        #     b[temp == l] = self.label_colours[l][2]
        #
        # rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        # rgb[:, :, 0] = r / 255.0
        # rgb[:, :, 1] = g / 255.0
        # rgb[:, :, 2] = b / 255.0
        # lbl = lbl[np.newaxis, ...]
        # lbl = np.transpose(lbl, [1, 2, 0]).astype('int32')
        # lbl = self.untf(lbl)
        lbl = Image.fromarray(lbl.astype('uint8'))
        lbl.putpalette(self.cityspallete)
        return lbl

    def encode_segmap(self, mask):
        # Put all void classes to 255
        for _voidc in self.void_classes:
            mask[mask == _voidc] = 255
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
