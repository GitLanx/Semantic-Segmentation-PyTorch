import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
parser.add_argument('--train_img_root', type=str, required=True, help='path to training images')
parser.add_argument('--train_lbl_root', type=str, required=True, help='path to training labels')
parser.add_argument('--val_img_root', type=str, help='path to validation images')
parser.add_argument('--val_lbl_root', type=str, help='path to validation labels')
parser.add_argument('--train_split', type=float, help='proportion of the dataset to include in the train split')

args = parser.parse_args()

train_img_root = args.train_img_root
train_lbl_root = args.train_lbl_root
val_img_root = args.val_img_root
val_lbl_root = args.val_lbl_root
train_split = args.train_split

if val_img_root is None:
    img = []
    lbl = []
    for root, _, files in os.walk(train_img_root):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img.append(os.path.join(root, filename))

    for root, _, files in os.walk(train_lbl_root):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                lbl.append(os.path.join(root, filename))

    assert len(img) == len(lbl), 'numbers of images and labels are not equal'

    choice = np.random.choice(len(img), len(img), replace=False)
    train = choice[:int(len(img) * train_split)]
    val = choice[int(len(img) * train_split):]

    with open('train.txt', 'a') as f:
        for index in train:
            f.write(' '.join([img[index], lbl[index]]) + '\n')

    with open('val.txt', 'a') as f:
        for index in val:
            f.write(' '.join([img[index], lbl[index]]) + '\n')
else:
    train_img = []
    train_lbl = []
    val_img = []
    val_lbl = []
    name_list = [train_img, train_lbl, val_img, val_lbl]
    root = [train_img_root, train_lbl_root, val_img_root, val_lbl_root]
    for nlist, root in zip(name_list, root):
        for _root, _, files in os.walk(root):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    nlist.append(os.path.join(_root, filename))

    with open('train.txt', 'a') as f:
        for index in range(len(train_img)):
            f.write(' '.join([train_img[index], train_lbl[index]]) + '\n')

    with open('val.txt', 'a') as f:
        for index in range(len(val_img)):
            f.write(' '.join([val_img[index], val_lbl[index]]) + '\n')
