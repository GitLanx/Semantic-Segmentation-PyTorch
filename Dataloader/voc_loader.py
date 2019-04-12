import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from .baseloader import BaseLoader


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

# Test code
# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     root = r'D:\Datasets\VOCdevkit\VOC2012'
#     batch_size = 2
#     loader = VOCLoader(root=root, img_size=(500, 500))
#     test_loader = DataLoader(loader, batch_size=batch_size, shuffle=True)

#     palette = test_loader.dataset.getpalette()
#     fig, axes = plt.subplots(batch_size, 2, subplot_kw={'xticks': [], 'yticks': []})
#     fig.subplots_adjust(left=0.03, right=0.97, hspace=0.2, wspace=0.05)

#     for imgs, labels in test_loader:
#         imgs = imgs.numpy()
#         imgs = np.transpose(imgs, [0,2,3,1])
#         labels = labels.numpy()

#         for i in range(batch_size):
#             axes[i][0].imshow(imgs[i])

#             mask_unlabeled = labels[i] == -1
#             viz_unlabeled = (
#                 np.zeros((labels[i].shape[0], labels[i].shape[1], 3))
#             ).astype(np.uint8)

#             lbl_viz = palette[labels[i]]
#             lbl_viz[labels[i] == -1] = (0, 0, 0)
#             lbl_viz[mask_unlabeled] = viz_unlabeled[mask_unlabeled]

#             axes[i][1].imshow(lbl_viz.astype(np.uint8))
#         plt.show()
#         break


