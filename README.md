# PyTorch models

## Dataset
### PASCAL VOC
model|acc|acc_cls|mean_iu|params
---|---|---|---|---
FCN32s<sup>1</sup>|89.00%|69.68%|56.49%|lr=1.0e-10<br>reduction='sum'
FCN32s<sup>2</sup>|90.17%|75.56%|61.81%|lr=1.0e-10<br>reduction='sum'
FCN32s(original)|-|-|63.6%|
FCN8sAtOnce<sup>1</sup>|89.30%|67.40%|56.51%|lr=1.0e-10<br>reduction='sum'
FCN8sAtOnce(original)|-|-|65.4%|
DeepLab-LargeFov<sup>1</sup>|89.10%|64.37%|54.19%|lr=0.001
DeepLab-MscLargeFov<sup>1</sup>|88.51%|67.04%|53.86%|lr=0.001
DeepLab-ASPP<sup>1</sup>|88.76%|71.07%|55.27%|lr=0.001
[1]train on voc2012, eval on voc2012
[2]train on sbd, eval on seg11valid

### CamVid
model|acc|acc_cls|mean_iu|params
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