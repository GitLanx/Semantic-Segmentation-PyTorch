import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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


# Test code
# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     root = r'D:/Datasets/CamVid'
#     batch_size = 2
#     loader = CamVidLoader(root=root, img_size=None)
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

