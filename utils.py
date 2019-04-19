import math
import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn
import skimage
import skimage.color
import skimage.transform
from torch.optim import lr_scheduler

# Adapted from https://github.com/wkentaro/fcn/blob/master/fcn/utils.py

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def centerize(src, dst_shape, margin_color=None):
    """Centerize image for specified image size
    @param src: image to centerize
    @param dst_shape: image shape (height, width) or (height, width, channel)
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    centerized[pad_vertical:pad_vertical + h, pad_horizontal:pad_horizontal +
               w] = src
    return centerized


def _tile_images(imgs, tile_shape, concatenated_image):
    """Concatenate images whose sizes are same.
    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    y_num, x_num = tile_shape
    one_width = imgs[0].shape[1]
    one_height = imgs[0].shape[0]
    if concatenated_image is None:
        if len(imgs[0].shape) == 3:
            n_channels = imgs[0].shape[2]
            assert all(im.shape[2] == n_channels for im in imgs)
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num, n_channels),
                dtype=np.uint8,
            )
        else:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num), dtype=np.uint8)
    for y in range(y_num):
        for x in range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y * one_height:(y + 1) * one_height, x *
                                   one_width:(x + 1) * one_width] = imgs[i]
    return concatenated_image


def get_tile_image(imgs, tile_shape=None, result_img=None, margin_color=None):
    """Concatenate images whose sizes are different.
    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param result_img: numpy array to put result image
    """

    def resize(*args, **kwargs):
        return skimage.transform.resize(*args, **kwargs)

    def get_tile_shape(img_num):
        x_num = 0
        y_num = int(math.sqrt(img_num))
        while x_num * y_num < img_num:
            x_num += 1
        return y_num, x_num

    if tile_shape is None:
        tile_shape = get_tile_shape(len(imgs))

    # get max tile size to which each image should be resized
    max_height, max_width = np.inf, np.inf
    for img in imgs:
        max_height = min([max_height, img.shape[0]])
        max_width = min([max_width, img.shape[1]])

    # resize and concatenate images
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        dtype = img.dtype
        h_scale, w_scale = max_height / h, max_width / w
        scale = min([h_scale, w_scale])
        h, w = int(scale * h), int(scale * w)
        img = resize(
            image=img,
            output_shape=(h, w),
            mode='reflect',
            preserve_range=True,
            anti_aliasing=True,
        ).astype(dtype)
        if len(img.shape) == 3:
            img = centerize(img, (max_height, max_width, 3), margin_color)
        else:
            img = centerize(img, (max_height, max_width), margin_color)
        imgs[i] = img
    return _tile_images(imgs, tile_shape, result_img)


def label2rgb(lbl, dataloader, img=None, n_labels=None, alpha=0.5):
    if n_labels is None:
        n_labels = lbl.max() + 1  # +1 for bg_label 0

    cmap = dataloader.dataset.getpalette()
    # cmap = getpalette(n_labels)
    # cmap = np.array(cmap).reshape([-1, 3]).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:

        # img_gray = skimage.color.rgb2gray(img)
        # img_gray = skimage.color.gray2rgb(img_gray)
        # img_gray *= 255
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz


def visualize_segmentation(**kwargs):
    """Visualize segmentation.
    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.
    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred = kwargs.pop('lbl_pred', None)
    n_class = kwargs.pop('n_classes', None)
    dataloader = kwargs.pop('dataloader', None)
    if kwargs:
        raise RuntimeError('Unexpected keys in kwargs: {}'.format(
            kwargs.keys()))

    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    mask_unlabeled = None
    viz_unlabeled = None
    if lbl_true is not None:
        mask_unlabeled = lbl_true == -1
        # lbl_true[mask_unlabeled] = 0
        viz_unlabeled = (np.zeros((lbl_true.shape[0], lbl_true.shape[1],
                                   3))).astype(np.uint8)
        # if lbl_pred is not None:
        #     lbl_pred[mask_unlabeled] = 0

    vizs = []

    if lbl_true is not None:
        viz_trues = [
            img,
            label2rgb(lbl_true, dataloader, n_labels=n_class),
            label2rgb(lbl_true, dataloader, img, n_labels=n_class),
        ]
        viz_trues[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_trues[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(get_tile_image(viz_trues, (1, 3)))

    if lbl_pred is not None:
        viz_preds = [
            img,
            label2rgb(lbl_pred, dataloader, n_labels=n_class),
            label2rgb(lbl_pred, dataloader, img, n_labels=n_class),
        ]
        if mask_unlabeled is not None and viz_unlabeled is not None:
            viz_preds[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
            viz_preds[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(get_tile_image(viz_preds, (1, 3)))

    if len(vizs) == 1:
        return vizs[0]
    elif len(vizs) == 2:
        return get_tile_image(vizs, (2, 1))
    else:
        raise RuntimeError


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

# Adapted from official CycleGAN implementation


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | poly | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':

        def lambda_rule(epoch):
            lr = 1.0 - max(0,
                           epoch + 1 - opt.epochs) / float(opt.niter_decay + 1)
            return lr

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'poly':

        def lambda_rule(epoch):
            lr = (1 - epoch / opt.epochs)**opt.lr_power
            return lr

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_step, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=1e-4, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)
    elif opt.lr_policy is None:
        scheduler = None
    else:
        return NotImplementedError(
            f'learning rate policy {opt.lr_policy} is not implemented')
    return scheduler


# Adapted from:
# https://github.com/wkentaro/pytorch-fcn/blob/master/examples/voc/learning_curve.py


def learning_curve(log_file):
    print(f'==> Plotting log file: {log_file}')

    df = pandas.read_csv(log_file)

    colors = ['red', 'green', 'blue', 'purple', 'orange']
    colors = seaborn.xkcd_palette(colors)

    plt.figure(figsize=(20, 6), dpi=300)

    row_min = df.min()
    row_max = df.max()

    # initialize DataFrame for train
    columns = [
        'epoch',
        'train/loss',
        'train/acc',
        'train/acc_cls',
        'train/mean_iu',
        'train/fwavacc',
    ]
    df_train = df[columns]
    # if hasattr(df_train, 'rolling'):
    #     df_train = df_train.rolling(window=10).mean()
    # else:
    #     df_train = pandas.rolling_mean(df_train, window=10)
    df_train = df_train.dropna()

    # initialize DataFrame for val
    columns = [
        'epoch',
        'valid/loss',
        'valid/acc',
        'valid/acc_cls',
        'valid/mean_iu',
        'valid/fwavacc',
    ]
    df_valid = df[columns]
    df_valid = df_valid.dropna()

    data_frames = {'train': df_train, 'valid': df_valid}

    n_row = 2
    n_col = 2
    for i, split in enumerate(['train', 'valid']):
        df_split = data_frames[split]

        # loss
        plt.subplot(n_row, n_col, i * n_col + 1)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(
            df_split['epoch'],
            df_split[f'{split}/loss'],
            '-',
            markersize=1,
            color=colors[0],
            alpha=.5,
            label=f'{split} loss')
        plt.xlim((1, row_max['epoch']))
        plt.ylim(
            min(df_split[f'{split}/loss']), max(df_split[f'{split}/loss']))
        plt.xlabel('epoch')
        plt.ylabel(f'{split} loss')

        # loss (log)
        # plt.subplot(n_row, n_col, i * n_col + 2)
        # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # plt.semilogy(df_split['epoch'], df_split[f'{split}/loss'],
        #              '-', markersize=1, color=colors[0], alpha=.5,
        #              label=f'{split} loss')
        # plt.xlim((1, row_max['epoch']))
        # plt.ylim(min(df_split[f'{split}/loss']), max(df_split[f'{split}/loss']))
        # plt.xlabel('epoch')
        # plt.ylabel('f{split} loss (log)')

        # lbl accuracy
        plt.subplot(n_row, n_col, i * n_col + 2)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(
            df_split['epoch'],
            df_split[f'{split}/acc'],
            '-',
            markersize=1,
            color=colors[1],
            alpha=.5,
            label=f'{split} accuracy')
        plt.plot(
            df_split['epoch'],
            df_split[f'{split}/acc_cls'],
            '-',
            markersize=1,
            color=colors[2],
            alpha=.5,
            label=f'{split} accuracy class')
        plt.plot(
            df_split['epoch'],
            df_split[f'{split}/mean_iu'],
            '-',
            markersize=1,
            color=colors[3],
            alpha=.5,
            label=f'{split} mean IU')
        plt.plot(
            df_split['epoch'],
            df_split[f'{split}/fwavacc'],
            '-',
            markersize=1,
            color=colors[4],
            alpha=.5,
            label=f'{split} fwav accuracy')
        plt.legend()
        plt.xlim((1, row_max['epoch']))
        plt.ylim((0, 1))
        plt.xlabel('epoch')
        plt.ylabel(f'{split} label accuracy')

    # out_file = osp.splitext(log_file)[0] + '.png'
    out_file = log_file[:-4] + '.png'
    plt.savefig(out_file)
    print(f'==> Wrote figure to: {out_file}')
