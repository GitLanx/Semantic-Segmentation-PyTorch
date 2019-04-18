import argparse

import numpy as np
from PIL import Image
import scipy
import torch
from torch.utils.data import DataLoader
import tqdm
import Models
from utils import visualize_segmentation, get_tile_image, runningScore, averageMeter
from Dataloader import get_loader
from augmentations import RandomCrop, Compose

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='deeplab-largefov')
    parser.add_argument('--model_file', type=str, default='/home/ecust/lx/Semantic-Segmentation-PyTorch/logs/deeplab-largefov_20190417_230357/model_best.pth.tar',help='Model path')
    parser.add_argument('--dataset_type', type=str, default='voc',help='type of dataset')
    parser.add_argument('--dataset', type=str, default='/home/ecust/Datasets/PASCAL VOC/VOCdevkit/VOC2012',help='path to dataset')
    parser.add_argument('--img_size', type=tuple, default=None, help='resize images using bilinear interpolation')
    parser.add_argument('--crop_size', type=tuple, default=None, help='crop images')
    parser.add_argument('--n_classes', type=int, default=21, help='number of classes')
    parser.add_argument('--pretrained', type=bool, default=True, help='should be set the same as train.py')
    args = parser.parse_args()

    model_file = args.model_file
    root = args.dataset
    n_classes = args.n_classes

    crop=None
    # crop = Compose([RandomCrop(args.crop_size)])
    loader = get_loader(args.dataset_type)
    val_loader = DataLoader(
        loader(root, n_classes=n_classes, split='val', img_size=args.img_size, augmentations=crop, pretrained=args.pretrained),
        batch_size=1, shuffle=False, num_workers=4)

    model, _, _ = Models.model_loader(args.model, n_classes, resume=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print('==> Loading {} model file: {}'.format(model.__class__.__name__, model_file))

    model_data = torch.load(model_file)

    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with {} dataset'.format(args.dataset_type))
    visualizations = []
    metrics = runningScore(n_classes)

    for data, target in tqdm.tqdm(val_loader, total=len(val_loader), ncols=80, leave=False):
        data, target = data.to(device), target.to(device)
        score = model(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            metrics.update(lt, lp)
            if len(visualizations) < 9:
                viz = visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img,
                    n_classes=n_classes, dataloader=val_loader)
                visualizations.append(viz)
    acc, acc_cls, mean_iu, fwavacc, cls_iu = metrics.get_scores()
    print('''
Accuracy:       {0:.2f}
Accuracy Class: {1:.2f}
Mean IoU:       {2:.2f}
FWAV Accuracy:  {3:.2f}'''.format(acc * 100,
                                  acc_cls * 100,
                                  mean_iu * 100,
                                  fwavacc * 100) + '\n')

    class_name = val_loader.dataset.class_names
    if class_name is not None:
        for index, value in enumerate(cls_iu.values()):
            offset = 20 - len(class_name[index])
            print(class_name[index] + ' ' * offset + f'{value * 100:>.2f}')
    else:
        print("\nyou don't specify class_names, use number instead")
        for key, value in cls_iu.items():
            print(key, f'{value * 100:>.2f}')

    viz = get_tile_image(visualizations)
    # img = Image.fromarray(viz)
    # img.save('viz_evaluate.png')
    scipy.misc.imsave('viz_evaluate.png', viz)

if __name__ == '__main__':
    main()