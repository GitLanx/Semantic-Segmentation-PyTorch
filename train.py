import os
import os.path as osp
import random
import yaml
import argparse
import datetime
import torch
from Dataloader import get_loader
from torch.utils.data import DataLoader
from Models import model_loader
from trainer import Trainer
from utils import get_scheduler
from optimizer import get_optimizer
from augmentations import get_augmentations

here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, default='deeplab-largefov', help='model to train for')
    parser.add_argument('--epochs', type=int, default=50, help='total epochs')
    parser.add_argument('--val_epoch', type=int, default=10, help='validation interval')
    parser.add_argument('--batch_size', type=int, default=16, help='number of batch size')
    parser.add_argument('--img_size', type=tuple, default=None, help='resize images to proper size')
    parser.add_argument('--dataset_type', type=str, default='voc', help='choose which dataset to use')
    parser.add_argument('--dataset_root', type=str, default='/home/ecust/Datasets/PASCAL VOC/VOC_Aug', help='path to dataset')
    parser.add_argument('--n_classes', type=int, default=21, help='number of classes')
    parser.add_argument('--resume', default=None, help='path to checkpoint')
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_policy', type=str, default='poly', help='learning rate policy')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum for sgd, beta1 for adam')
    parser.add_argument('--lr_decay_step', type=float, default=10, help='step size for step learning policy')
    parser.add_argument('--lr_power', type=int, default=0.9, help='power parameter for poly learning policy')
    parser.add_argument('--pretrained', type=bool, default=True, help='whether to use pretrained models')
    parser.add_argument('--iter_size', type=int, default=10, help='iters to accumulate gradients')

    parser.add_argument('--crop_size', type=tuple, default=(321, 321), help='crop sizes of images')
    parser.add_argument('--flip', type=bool, default=True, help='whether to use horizontal flip')

    args = parser.parse_args()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', args.model + '_' + now.strftime('%Y%m%d_%H%M%S'))

    if not osp.exists(args.out):
        os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Start training {args.model} using {device.type}\n')

    random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    # 1. dataset

    root = args.dataset_root
    loader = get_loader(args.dataset_type)

    augmentations = get_augmentations(args)

    train_loader = DataLoader(
        loader(root, n_classes=args.n_classes, split='train_aug', img_size=args.img_size, augmentations=augmentations,
               pretrained=args.pretrained),
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        loader(root, n_classes=args.n_classes, split='val_id', img_size=args.img_size, pretrained=args.pretrained),
        batch_size=1, shuffle=False, num_workers=4)

    # 2. model
    model, start_epoch, ckpt = model_loader(args.model, args.n_classes, args.resume)
    model = model.to(device)

    # 3. optimizer
    optim = get_optimizer(args, model)
    if args.resume:
        optim.load_state_dict(ckpt['optim_state_dict'])

    scheduler = get_scheduler(optim, args)

    # 4. train
    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        epochs=args.epochs,
        n_classes=args.n_classes,
        val_epoch=args.val_epoch,
        iter_size=args.iter_size
    )
    trainer.epoch = start_epoch
    trainer.train()


if __name__ == '__main__':
    main()
