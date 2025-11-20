import torch
from PIL import Image
import numpy as np
import tifffile as tf
import gdal
from torch.nn import init
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision import models
from sklearn.metrics import confusion_matrix
# from segmetric import metric

# a = [[0, 1, 2, 3], [1, 2, 3, 4]]
# print(np.mean(a, axis=0))
a = np.array([[1, 2],
              [2, 1]])
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if a[i, j] == 0:
            print('0')
        if a[i, j] == 1:
            print('1')
        if a[i, j] == 2:
            print('2')
# for i in range(4):
#     a.append(i)
# print(a)
# print(np.mean(a))
# class FeatureResNet(ResNet):
#     def __init__(self):
#         super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
#
#     def forward(self, x):
#
#         x1 = self.conv1(x)
#         x = self.bn1(x1)
#         x = self.relu(x)
#         x2 = self.maxpool(x)
#         x = self.layer1(x2)
#         x3 = self.layer2(x)
#         x4 = self.layer3(x3)
#         x5 = self.layer4(x4)
#
#         return x1, x2, x3, x4, x5
#
# pretrained_net = FeatureResNet()
# pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())
# pretrained_net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# print(pretrained_net)

# imglabel = np.array([[0, 1, 2],
#                      [2, 0, 0],
#                      [0, 1, -1]])
# imgpre = np.array([[0, 1, 2],
#                      [1, 2, 0],
#                      [0, 1, -1]])
# # acc = np.diag(imgpre).sum()
# # print(acc)
# numclass = 3
# mask = (imglabel >= 0) & (imglabel<numclass)
# print(mask)
# print(numclass*imglabel[mask].astype(int)+imgpre[mask])
# label = numclass * imglabel[mask].astype(int) + imgpre[mask]
# print(label)
# count = np.bincount(label, minlength=numclass**2)
# print(count)
# confusionmatrix = count.reshape(numclass, numclass)
# print(confusionmatrix)
#
# # imglabel = imglabel.reshape(-1)
# # imgpre = imgpre.reshape(-1)
# # a = confusion_matrix(imglabel, imgpre)
# # print(a)
# segmetric = metric(num_class=3)
# segmetric.addBatch(imgpre, imglabel)
# AA = segmetric.CPA()
# print(AA)