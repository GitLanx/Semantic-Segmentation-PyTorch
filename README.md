# PyTorch models

## Dataset
### PASCAL VOC
model|acc|acc_cls|mean_iu|params
---|---|---|---|---
FCN32s<sup>1</sup>|89.37%|68.53%|56.93%|lr=1.0e-10<br>reduction='sum'
FCN32s<sup>2</sup>|90.22%|74.44%|62.01%|lr=1.0e-10<br>reduction='sum'
FCN32s(original)|-|-|63.6%|
FCN8sAtOnce<sup>1</sup>|89.55%|68.98%|57.37%|lr=1.0e-10<br>reduction='sum'
FCN8sAtOnce(original|-|-|65.4%|
[1]train on voc2012, eval on voc2012  
[2]train on sbd, eval on seg11valid

### CamVid
model|acc|acc_cls|mean_iu|params
---|---|---|---|---
SegNet(Maxunpooling)|86.96%|61.47%|50.61%|lr=0.1<br>reduction='mean'
SegNet(original)|88.6%|65.9%|50.2%
UNet|85.17%|59.79%|48.70%|lr=0.01<br>reduction='mean'
DeepLab-LargeFov|85.97%|62.76%|50.84%|lr=0.01<br>reduction='mean'
DeepLab-MscLargeFov|88.23%|65.12%|55.26%|lr=0.01<br>reduction='mean'
DeepLab-ASPP|
fcn32s|85.71%|59.66%|49.34%|lr=0.01<br>reduction='mean'