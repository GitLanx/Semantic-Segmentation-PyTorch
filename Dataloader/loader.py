import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils import data
from torchvision import transforms


class BaseLoader(data.Dataset):
    '''Adapted from:
    https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
    https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/datasets/voc.py
    '''
    def __init__(
        self,
        root,
        split="train",
        transform=False,
        img_size=None,
        augmentations=None,
        n_classes=14,
        ignore_index=None,
    ):
        self.root = root
        self.split = split
        self.is_transform = transform
        self.augmentations = augmentations
        self.img_size = img_size
        self.ignore_index = ignore_index

        path = os.path.join(self.root, split + ".txt")
        with open(path, "r") as f:
            self.file_list = [file_name.rstrip().split() for file_name in f]

        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index][0]
        lbl_name = self.file_list[index][1]

        img = Image.open(os.path.join(self.root, img_name))
        lbl = Image.open(os.path.join(self.root, lbl_name))

        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def transform(self, img, lbl):
        img = self.tf(img)
        lbl = np.array(lbl, dtype=np.int32)
        if self.ignore_index is not None:
            lbl[lbl == self.ignore_index] = 255

        lbl = torch.from_numpy(lbl).long()
        return img, lbl

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
        for ii, label in enumerate(self.get_pascal_labels()):
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
        # label_colours = self.get_pascal_labels()
        # r = label_mask.copy()
        # g = label_mask.copy()
        # b = label_mask.copy()
        # for ll in range(0, self.n_classes):
        #     r[label_mask == ll] = label_colours[ll, 0]
        #     g[label_mask == ll] = label_colours[ll, 1]
        #     b[label_mask == ll] = label_colours[ll, 2]
        # rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        # rgb[:, :, 0] = r / 255.0
        # rgb[:, :, 1] = g / 255.0
        # rgb[:, :, 2] = b / 255.0
        # if plot:
        #     plt.imshow(rgb)
        #     plt.show()
        # else:
        #     return rgb
        return label_mask


# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
if __name__ == '__main__':
    local_path = '/home/ecust/lx/数据库A(200)'
    n_classes = 14
    bs = 4
    dst = BaseLoader(root=local_path, transform=True)
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]), vmin=0, vmax=n_classes)
        plt.show()

