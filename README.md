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
SegNet(Maxunpooling)|87.44%|72.16%|58.96%|lr=0.01
SegNet(Bilinear interpolation)|85.86%|71.95%|56.22%|lr=0.01
SegNet(original)|88.6%|65.9%|50.2%
UNet|84.38%|62.80%|49.83%|lr=0.01
DeepLab-LargeFov|85.97%|62.76%|50.84%|lr=0.01<br>reduction='mean'
DeepLab-MscLargeFov|86.07%|70.73%|54.51%|lr=0.01<br>reduction='mean'
DeepLab-ASPP|85.67%|72.87%|55.13%|lr=0.01
DeepLab-v3|79.77%|67.84%|46.94%|
fcn32s|85.71%|59.66%|49.34%|lr=0.01<br>reduction='mean'