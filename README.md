# PyTorch models

## Dataset
### PASCAL VOC
model|acc|acc_cls|mean_iu|notes
---|---|---|---|---
FCN32s|90.17%|75.56%|61.81%|lr=1.0e-10<br>reduction='sum'
FCN32s(original)|-|-|63.6%|
FCN8sAtOnce|90.27%|74.95%|62.13%|lr=1.0e-10<br>reduction='sum'
FCN8sAtOnce(original)|-|-|65.4%|
DeepLab-LargeFov|93.79%|72.67%|61.68%|pad images to 513x513 for evaluation
DeepLab-LargeFov|91.02%|74.54%|62.58%|use original resolution for evaluation
DeepLab-LargeFov(original)|-|-|62.25%|
DeepLab-ASPP<sup>1</sup>|88.76%|71.07%|55.27%|lr=0.001
[1]train on voc2012, eval on voc2012

### CamVid
model|acc|acc_cls|mean_iu|notes
---|---|---|---|---
SegNet(Maxunpooling, vgg16-based)|86.71%|66.39%|54.09%|lr=0.01
SegNet(Maxunpooling, vg16_bn-based)|87.84%|70.75%|57.68%|lr=0.01
SegNet(Bilinear interpolation)|85.86%|71.95%|56.22%|lr=0.01
SegNet(original)|88.6%|65.9%|50.2%
UNet|84.38%|62.80%|49.83%|lr=0.01
