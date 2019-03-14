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

here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, default='unet', help='model to train for')
    parser.add_argument('--epochs', type=int, default=200, help='total epochs')
    parser.add_argument('--val_epoch', type=int, default=10, help='validation interval')
    parser.add_argument('--batch_size', type=int, default=4, help='number of batch size')
    parser.add_argument('--img_size', type=tuple, default=(128, 128), help='resize images to proper size')
    parser.add_argument('--dataset_type', type=str, default='camvid', help='choose which dataset to use')
    parser.add_argument('--train_root', type=str, default='D:/Datasets/CamVid', help='path to train.txt')
    parser.add_argument('--val_root', type=str, help='path to val.txt')
    parser.add_argument('--n_classes', type=int, default=11, help='number of classes')
    parser.add_argument('--resume', help='path to checkpoint')
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum for sgd, beta1 for adam')

    args = parser.parse_args()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', args.model + '_' + now.strftime('%Y%m%d_%H%M%S'))

    if not osp.exists(args.out):
        os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    # 1. dataset

    root = args.train_root
    # if args.dataset_type == '':
    #     from Dataloader import BaseLoader as loader
    # elif args.dataset_type == 'voc':
    #     from Dataloader import VOCLoader as loader
    loader = get_loader(args.dataset_type)

    train_loader = DataLoader(
        loader(root, split='train', transform=True, img_size=args.img_size),
        batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        loader(root, split='val', transform=True, img_size=args.img_size),
        batch_size=1, shuffle=False, num_workers=4)

    # 2. model
    model, start_epoch, ckpt = model_loader(args.model, args.n_classes, args.resume)
    model = model.to(device)

    # 3. optimizer
    if args.optim.lower() == 'sgd':
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.beta1,
            weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(ckpt['optim_state_dict'])

    # 4. train
    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        epochs=args.epochs,
        n_classes=args.n_classes,
        val_epoch=args.val_epoch,
    )
    trainer.epoch = start_epoch
    trainer.train()


if __name__ == '__main__':
    main()
