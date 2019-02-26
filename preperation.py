import os
import numpy as np

img_root = '/home/ecust/lx/数据库A(200)/IR_400x300'
lbl_root = '/home/ecust/lx/数据库A(200)/label_png'

img = []
lbl = []
for root, _, files in os.walk(img_root):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img.append(os.path.join(root.split(os.sep)[-1], file))

for root, _, files in os.walk(lbl_root):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            lbl.append(os.path.join(root.split(os.sep)[-1], file))

assert len(img) == len(lbl), 'numbers of images and labels are not equal'

choice = np.random.choice(len(img), len(img), replace=False)
train = choice[:int(len(img) * 0.8)]
val = choice[int(len(img) * 0.8):]

with open('train.txt', 'a') as f:
    for index in train:
        f.write(' '.join([img[index], lbl[index]]) + '\n')

with open('val.txt', 'a') as f:
    for index in val:
        f.write(' '.join([img[index], lbl[index]]) + '\n')
