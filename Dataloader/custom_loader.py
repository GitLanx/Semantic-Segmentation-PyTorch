import os
import numpy as np
import matplotlib.pyplot as plt
from .baseloader import BaseLoader
from PIL import Image


class CustomLoader(BaseLoader):
    """Custom dataset loader.
    Parameters
    ----------
      root: path to custom dataset, with train.txt and val.txt together.
        i.e., -----dataset
                |--train.txt
                |--val.txt
      n_classes: number of classes.
      split: choose subset of dataset, 'train','val' or 'test'.
      img_size: scale image to proper size.
      augmentations: whether to perform augmentation.
      ignore_index: ingore_index will be ignored in training phase and evaluation.
      class_weight: useful in unbalanced datasets.
      pretrained: whether to use pretrained models
    """
    # specify class_names if necessary
    class_names = None

    def __init__(
        self,
        root,
        n_classes,
        split="train",
        img_size=None,
        augmentations=None,
        ignore_index=None,
        class_weight=None,
        pretrained=False
        ):
        super(CustomLoader, self).__init__(root, n_classes, split, img_size, augmentations, ignore_index, class_weight, pretrained)

        path = os.path.join(self.root, split + ".txt")
        with open(path, "r") as f:
            self.file_list = [file_name.rstrip().split() for file_name in f]

        print(f"Found {len(self.file_list)} {split} images")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index][0]
        lbl_name = self.file_list[index][1]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        lbl = Image.open(os.path.join(self.root, lbl_name))

        if self.img_size:
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)

        if self.augmentations:
            img, lbl = self.augmentations(img, lbl)

        img, lbl = self.transform(img, lbl)
        return img, lbl

    def getpalette(self):
        """for custom palette, if not specified, use pascal voc palette by default.
        """
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
#     root = ''
#     batch_size = 2
#     loader = CustomLoader(root=root, img_size=None)
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

