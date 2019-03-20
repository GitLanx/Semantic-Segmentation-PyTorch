import argparse

import numpy as np
from PIL import Image
import scipy
import torch
from torch.utils.data import DataLoader
import tqdm
import Models
from utils import visualize_segmentation, label_accuracy_score, get_tile_image
from Dataloader import get_loader

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='pspnet')
    parser.add_argument('--model_file', type=str, default='D:/lx/Semantic-Segmentation-PyTorch/logs/pspnet_20190320_172958/model_best.pth.tar',help='Model path')
    parser.add_argument('--dataset_type', type=str, default='camvid',help='type of dataset')
    parser.add_argument('--dataset', type=str, default='D:/Datasets/CamVid',help='path to dataset')
    parser.add_argument('--img_size', type=tuple, default=(321, 321), help='resize images')
    args = parser.parse_args()

    model_file = args.model_file
    root = args.dataset

    loader = get_loader(args.dataset_type)
    val_loader = DataLoader(
        loader(root, split='test', transform=True, img_size=args.img_size),
        batch_size=1, shuffle=False, num_workers=4)

    n_classes = val_loader.dataset.n_classes

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
    label_trues, label_preds = [], []
    for data, target in tqdm.tqdm(val_loader, total=len(val_loader), ncols=80, leave=False):
        data, target = data.to(device), target.to(device)
        score = model(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                viz = visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_classes=n_classes)
                visualizations.append(viz)
    metrics = label_accuracy_score(
        label_trues, label_preds, n_class=n_classes)
    metrics = np.array(metrics)
    metrics *= 100
    print('''Accuracy: {0}
             Accuracy Class: {1}
             Mean IoU: {2}
             FWAV Accuracy: {3}'''.format(*metrics))

    viz = get_tile_image(visualizations)
    # img = Image.fromarray(viz)
    # img.save('viz_evaluate.png')
    scipy.misc.imsave('viz_evaluate.png', viz)

if __name__ == '__main__':
    main()