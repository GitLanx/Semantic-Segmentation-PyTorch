import os
import collections
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .baseloader import BaseLoader

class CamVidLoader(BaseLoader):
    """CamVid dataset loader.
    Parameters
    ----------
      root: path to CamVid dataset.
      n_classes: number of classes, default 11.
      split: choose subset of dataset, 'train','val' or 'test'.
      img_size: a list or a tuple, scale image to proper size.
      augmentations: whether to perform augmentation.
      ignore_index: ingore_index will be ignored in training phase and evaluation, default 11.
      class_weight: useful in unbalanced datasets.
      pretrained: whether to use pretrained models
    """
    class_names = np.array([
        'sky',
        'building',
        'pole',
        'road',
        'pavement',
        'tree',
        'sign',
        'fence',
        'vehicle',
        'pedestrian',
        'bicyclist',
        'void'
    ])

    def __init__(
        self,
        root,
        n_classes=11,
        split='train',
        img_size=None,
        augmentations=None,
        ignore_index=11,
        class_weight=None,
        pretrained=False
        ):
        super(CamVidLoader, self).__init__(root, n_classes, split, img_size, augmentations, ignore_index, class_weight, pretrained)

        path = os.path.join(self.root, self.split + ".txt")
        with open(path, "r") as f:
            self.file_list = [file_name.rstrip() for file_name in f]

        self.class_weight = [0.2595, 0.1826, 4.5640, 0.1417,
                             0.9051, 0.3826, 9.6446, 1.8418,
                             0.6823 ,6.2478, 7.3614]

        print(f"Found {len(self.file_list)} {split} images")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_name = img_name.split()[0].split('/')[-1]
        img_path = os.path.join(self.root, self.split, img_name)
        if self.split == 'train':
            lbl_path = os.path.join(self.root, 'trainannot', img_name)
        elif self.split == 'val':
            lbl_path = os.path.join(self.root, 'valannot', img_name)
        elif self.split == 'test':
            lbl_path = os.path.join(self.root, 'testannot', img_name)

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
        return np.asarray(
            [
                [128, 128, 128],
                [128, 0, 0],
                [192, 192, 128],
                [128, 64, 128],
                [0, 0, 192],
                [128, 128, 0],
                [192, 128, 128],
                [64, 64, 128],
                [64, 0, 128],
                [64, 64, 0],
                [0, 128, 192],
            ]
        )

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

    def decode_segmap(self, label_mask):
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

# Leave code for debugging purposes
if __name__ == '__main__':
    root = r'D:/Datasets/CamVid'
    batch_size = 4
    loader = CamVidLoader(root=root, img_size=(320, 320))
    test_loader = DataLoader(loader, batch_size=batch_size)
    for imgs, labels in enumerate(test_loader):
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        labels = labels.numpy()

        fig, axes = plt.subplots(batch_size, 2)
        for i in range(batch_size):
            axes[i][0].imshow(imgs[i])

            mask_unlabeled = labels[i] == -1
            viz_unlabeled = (
                np.zeros((labels[i].shape[1], labels[i].shape[2], 3))
            ).astype(np.uint8)
            palette = loader.dataset.getpalette()
            lbl_viz = palette[labels[i]]
            lbl_viz[labels[i] == -1] = (0, 0, 0)
            lbl_viz[i][mask_unlabeled] = viz_unlabeled[mask_unlabeled]

            axes[i][1].imshow(lbl_viz[i])
        plt.show()
        break

