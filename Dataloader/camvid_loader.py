import os
import collections
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import transforms


class CamVidLoader(data.Dataset):
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
        split="train",
        transform=False,
        img_size=None,
        augmentations=None
    ):
        self.root = root
        self.split = split
        self.is_transform = transform
        self.augmentations = augmentations
        self.n_classes = 11
        self.img_size = img_size

        path = os.path.join(self.root, self.split + ".txt")
        with open(path, "r") as f:
            self.file_list = [file_name.rstrip() for file_name in f]

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean.tolist(), self.std.tolist()),
            ]
        )
        self.untf = transforms.Compose(
            [
                transforms.Normalize((-self.mean / self.std).tolist(),
                                     (1.0 / self.std).tolist()),
            ]
        )

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
        lbl[lbl == 11] = -1
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = self.untf(img)
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img * 255
        img = img.astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl

    def get_label_colors(self):
        """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
        """
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
                [0, 0, 0],
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
        for ii, label in enumerate(self.get_label_colors()):
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
        label_colours = self.get_label_colors()
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

