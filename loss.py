import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch


def CrossEntropyLoss(score, target, weight, ignore_index, reduction):
    """Cross entropy for single or multiple outputs.
    """
    if not isinstance(score, tuple):
        loss = F.cross_entropy(
            score, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
        return loss

    loss = 0
    for s in score:
        loss = loss + F.cross_entropy(
            s, target, weight=weight, ignore_index=ignore_index, reduction=reduction)
    return loss

def resize_labels(labels, size):
    new_labels = []
    for label in labels:
        label = label.float().cpu().numpy()
        label = Image.fromarray(label).resize((size[1], size[0]), Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels