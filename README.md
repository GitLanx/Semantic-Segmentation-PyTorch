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
DeepLab-LargeFov|84.19%|69.25%|52.13%|lr=0.01
DeepLab-MscLargeFov|84.85%|75.97%|55.23%|lr=0.01
DeepLab-ASPP|85.79%|73.77%|55.49%|lr=0.01
DeepLab-v3|85.86%|67.20%|53.24%|lr=0.01
PSPNet|83.55%|61.13%|48.78%|lr=0.01
fcn32s|82.04%|66.38%|48.62%|lr=0.01<br>reduction='mean'
fcn8s|83.62%|66.79%|50.36%|lr=0.01<br>reduction='mean'