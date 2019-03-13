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
SegNet|88.49%|60.07%|50.07%|lr=0.1<br>reduction='mean'
SegNet(original)|88.6%|65.9%|50.2%
UNet|90.70%|72.76%|60.65%|lr=0.001<br>reduction='mean'