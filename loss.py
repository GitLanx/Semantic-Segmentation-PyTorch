import torch.nn.functional as F


def CrossEntropyLoss(score, target, weight, ignore_index, reduction):
    if not isinstance(score, tuple):
        loss = F.cross_entropy(
            score, target, weight=weight, ignore_index=-1, reduction='mean')
        return loss

    loss = 0
    for s in score:
        target = F.interpolate(
            target, size=s.size[1:], mode='nearest', align_corners=True)
        loss = loss + F.cross_entropy(
            score, target, weight=weight, ignore_index=-1, reduction='mean')
    return loss
