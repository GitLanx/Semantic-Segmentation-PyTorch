import os
import collections
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from .baseloader import BaseLoader
import torch
from torch.utils import data
from torchvision import transforms


class VOCLoader(BaseLoader):
    """PASCAL VOC dataset loader.
    Parameters
    ----------
      root: path to pascal voc dataset.
        for directory:
        --VOCdevkit--VOC2012---ImageSets
                             |-JPEGImages
                             |-   ...
        root should be xxx/VOCdevkit/VOC2012
      n_classes: number of classes, default 21.
      split: choose subset of dataset, 'train','val' or 'trainval'.
      img_size: scale image to proper size.
      augmentations: whether to perform augmentation.
      ignore_index: ingore_index will be ignored in training phase and evaluation, default 255.
      class_weight: useful in unbalanced datasets.
      pretrained: whether to use pretrained models
    """
    class_names = np.array([
        'background', 'aeroplane', 'bicycle',
        'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person',
        'potted plant', 'sheep', 'sofa', 'train',
        'tv/monitor',
    ])

    def __init__(
        self,
        root,
        n_classes=21,
        split="train",
        img_size=None,
        augmentations=None,
        ignore_index=255,
        class_weight=None,
        pretrained=False
        ):
        super(VOCLoader, self).__init__(root, n_classes, split, img_size, augmentations, ignore_index, class_weight, pretrained)

        path = os.path.join(self.root, "ImageSets/Segmentation", split + ".txt")
        with open(path, "r") as f:
            self.file_list = [file_name.rstrip() for file_name in f]

        print(f"Found {len(self.file_list)} {split} images")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, "JPEGImages", img_name + ".jpg")
        lbl_path = os.path.join(self.root, "SegmentationClass", img_name + ".png")
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)
        if self.img_size:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        if self.augmentations:
            img, lbl = self.augmentations(img, lbl)

        img, lbl = self.transform(img, lbl)
        return img, lbl

    def getpalette(self):
        n = self.n_classes
        palette = [0]*(n*3)
        for j in range(0, n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
        palette = np.array(palette).reshape([-1, 3]).astype(np.uint8)
        return palette

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.getpalette()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
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

class SBDLoader(BaseLoader):
    """Semantic Boundaries Dataset(SBD) dataset loader.
    Parameters
    ----------
      root: path to SBD dataset.
        for directory:
        --benchmark_RELEASE--dataset---img
                                     |-cls
                                     |-train.txt
                                     |-  ...
        root should be xxx/benchmark_RELEASE
      n_classes: number of classes, default 21.
      split: choose subset of dataset, 'train' or 'val'.
      img_size: scale image to proper size.
      augmentations: whether to perform augmentation.
      ignore_index: ingore_index will be ignored in training phase and evaluation, default 255.
      class_weight: useful in unbalanced datasets.
      pretrained: whether to use pretrained models
    """
    class_names = np.array([
        'background', 'aeroplane', 'bicycle',
        'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person',
        'potted plant', 'sheep', 'sofa', 'train',
        'tv/monitor',
    ])
    def __init__(
        self,
        root,
        n_classes=21,
        split="train",
        img_size=None,
        augmentations=None,
        ignore_index=255,
        class_weight=None,
        pretrained=False
        ):
        super(SBDLoader, self).__init__(root, n_classes, split, img_size, augmentations, ignore_index, class_weight, pretrained)

        path = os.path.join(self.root, 'dataset', split + ".txt")
        with open(path, "r") as f:
            self.file_list = [file_name.rstrip() for file_name in f]

        print(f"Found {len(self.file_list)} {split} images")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, 'dataset/img', img_name + '.jpg')
        lbl_path = os.path.join(self.root, 'dataset/cls', img_name + '.mat')

        img = Image.open(img_path).convert('RGB')
        lbl = loadmat(lbl_path)
        lbl = lbl['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl = Image.fromarray(lbl)
        if self.img_size:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        if self.augmentations:
            img, lbl = self.augmentations(img, lbl)

        img, lbl = self.transform(img, lbl)
        return img, lbl

    def getpalette(self):
        n = self.n_classes
        palette = [0]*(n*3)
        for j in range(0, n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
        palette = np.array(palette).reshape([-1, 3]).astype(np.uint8)
        return palette

class VOC11Val(BaseLoader):
    """load PASCAL VOC 2012 dataset, but only use seg11valid.txt for evaluation.
    Parameters
    ----------
      root: path to PASCAL VOC 2012 dataset.
      n_classes: number of classes, default 21.
      split: only 'seg11valid' is available.
      img_size: scale image to proper size.
      augmentations: whether to perform augmentation.
      ignore_index: ingore_index will be ignored in training phase and evaluation, default 255.
      class_weight: useful in unbalanced datasets.
      pretrained: whether to use pretrained models
    """
    def __init__(
        self,
        root,
        n_classes=21,
        split="seg11valid",
        img_size=None,
        augmentations=None,
        ignore_index=255,
        class_weight=None,
        pretrained=False
        ):
        super(VOC11Val, self).__init__(root, n_classes, split, img_size, augmentations, ignore_index, class_weight, pretrained)

        current_path = os.path.realpath(__file__)

        path = os.path.join(current_path[:-13] + "seg11valid.txt")
        with open(path, "r") as f:
            self.file_list = [file_name.rstrip() for file_name in f]

        print(f"Found {len(self.file_list)} {split} images")

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, "JPEGImages", img_name + ".jpg")
        lbl_path = os.path.join(self.root, "SegmentationClass", img_name + ".png")
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)

        if self.img_size:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        if self.augmentations:
            img, lbl = self.augmentations(img, lbl)

        img, lbl = self.transform(img, lbl)
        return img, lbl

# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
if __name__ == '__main__':
    local_path = r'E:\dataset\VOC2012'
    bs = 4
    # augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
    dst = VOCLoader(root=local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()

