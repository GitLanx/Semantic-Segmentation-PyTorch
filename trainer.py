import datetime
import os
import os.path as osp
import shutil
from utils import label_accuracy_score, visualize_segmentation, get_tile_image
import numpy as np
import pytz
import scipy.misc
import torch
import torch.nn.functional as F
import tqdm


class Trainer:
    def __init__(self, device, model, optimizer, scheduler, train_loader, val_loader,
                 out, epochs, n_classes, val_epoch=None):
        self.device = device

        self.model = model
        self.optim = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('UTC'))

        self.val_epoch = val_epoch

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.n_classes = n_classes
        self.epoch = 0
        self.epochs = epochs
        self.best_mean_iu = 0

    def train_epoch(self):
        self.model.train()

        label_trues, label_preds = [], []
        for data, target in tqdm.tqdm(
                self.train_loader, total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            assert self.model.training

            data, target = data.to(self.device), target.to(self.device)
            self.optim.zero_grad()
            score = self.model(data)

            weight = self.train_loader.dataset.class_weight
            if weight:
                weight = torch.Tensor(weight).to(self.device)

            loss = F.cross_entropy(score, target, weight=weight, reduction='mean', ignore_index=-1)
            # loss /= len(data)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            # metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()
            lbl_true = target.data.cpu().numpy()
            for lt, lp in zip(lbl_true, lbl_pred):
                label_trues.append(lt)
                label_preds.append(lp)

            metrics = label_accuracy_score(
                    label_trues, label_preds, self.n_classes)

            # metrics.append((acc, acc_cls, mean_iu, fwavacc))
            # metrics = np.mean(metrics, axis=0)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('UTC')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch] + [loss_data] + \
                list(metrics) + [''] * 5 + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')
        
        if self.scheduler:
            self.scheduler.step()

        if self.epoch % self.val_epoch == 0:
            self.validate()
            lr = self.optim.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

    def validate(self):

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        with torch.no_grad():
            self.model.eval()
            for data, target in tqdm.tqdm(
                    self.val_loader, total=len(self.val_loader),
                    desc=f'Valid epoch={self.epoch}', ncols=80, leave=False):

                data, target = data.to(self.device), target.to(self.device)

                score = self.model(data)

                loss = F.cross_entropy(score, target, reduction='sum', ignore_index=-1)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data / len(data)

                imgs = data.data.cpu()
                lbl_pred = score.data.max(1)[1].cpu().numpy()
                lbl_true = target.data.cpu()
                for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                    img, lt = self.val_loader.dataset.untransform(img, lt)
                    label_trues.append(lt)
                    label_preds.append(lp)
                    if len(visualizations) < 9:
                        viz = visualize_segmentation(
                            lbl_pred=lp, lbl_true=lt, img=img,
                            n_classes=self.n_classes)
                        visualizations.append(viz)
        metrics = label_accuracy_score(
            label_trues, label_preds, self.n_classes)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'epoch%08d.jpg' % self.epoch)
        scipy.misc.imsave(out_file, get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('UTC')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.epochs + 1,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
