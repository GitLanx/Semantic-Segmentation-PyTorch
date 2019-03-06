import os
import collections
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
from torch.utils import data
from torchvision import transforms


class VOCLoader(data.Dataset):
    '''Adapted from:
    https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
    https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/datasets/voc.py
    '''
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
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
        self.n_classes = 21
        self.files = collections.defaultdict(list)
        self.img_size = img_size

        path = os.path.join(self.root, "ImageSets/Segmentation", split + ".txt")
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
        img_path = os.path.join(self.root, "JPEGImages", img_name + ".jpg")
        lbl_path = os.path.join(self.root, "SegmentationClass", img_name + ".png")
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
        lbl[lbl == 255] = -1
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
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
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
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

class SBDLoader(VOCLoader):
    def __init__(
        self,
        root,
        split='train',
        transform=False,
        img_size=None,
        augmentations=None
    ):
        self.root = root
        self.split = split
        self.is_transform = transform
        self.augmentations = augmentations
        self.n_classes = 21
        self.files = collections.defaultdict(list)
        self.img_size = img_size

        path = os.path.join(self.root, 'dataset', split + ".txt")
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

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, 'dataset/img', img_name + '.jpg')
        lbl_path = os.path.join(self.root, 'dataset/cls', img_name + '.mat')

        img = Image.open(img_path).convert('RGB')
        lbl = loadmat(lbl_path)
        lbl = lbl['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl = Image.fromarray(lbl)
        if self.img_size is not None:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl


class VOC11Val(VOCLoader):
    def __init__(
        self,
        root,
        split='val',
        transform=False,
        img_size=None,
        augmentations=None
    ):
        self.root = root
        self.split = split
        self.is_transform = transform
        self.augmentations = augmentations
        self.n_classes = 21
        self.img_size = img_size

        current_path = os.path.realpath(__file__)

        path = os.path.join(current_path[:-13] + "seg11valid.txt")
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

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.root, "JPEGImages", img_name + ".jpg")
        lbl_path = os.path.join(self.root, "SegmentationClass", img_name + ".png")
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

