import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

class BaseLoader(data.Dataset):
    def __init__(
        self,
        root,
        n_classes,
        split='train',
        img_size=None,
        augmentations=None,
        ignore_index=None,
        class_weight=None,
        pretrained=False
        ):
        self.root = root
        self.n_classes = n_classes
        self.split = split
        self.img_size = img_size
        self.augmentations = augmentations
        self.ignore_index = ignore_index
        self.class_weight = class_weight

        if pretrained:
            self.mean = torch.tensor([0.485, 0.456, 0.406])
            self.std = torch.tensor([0.229, 0.224, 0.225])
            self.tf = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(self.mean.tolist(), self.std.tolist())]
            )
            self.untf = transforms.Compose(
                [transforms.Normalize((-self.mean / self.std).tolist(),
                                    (1.0 / self.std).tolist())]
            )
        else:
            self.tf = transforms.Compose(
            [transforms.ToTensor()]
            )
            self.untf = transforms.Compose(
                [transforms.Normalize([0, 0, 0], [1, 1, 1])]
            )

    def __getitem__(self, index):
        return NotImplementedError

    def transform(self, img, lbl):
        img = self.tf(img)
        lbl = np.array(lbl, dtype=np.int32)
        if self.ignore_index:
            lbl[lbl == self.ignore_index] = -1
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
    
    def getpalette(self):
        return NotImplementedError