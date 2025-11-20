#语义分割进行度量
#利用混淆矩阵计算精度
#  TP(1被认为是1）  FP（0被认为是1）
#  FN(1被认为是0）  TN（0被认为是0）
#指标有
#OA/PA 像素准确率: （TP+TN)/(TP+FN+FP+TN)
#OA of per class/CPA 类别像素准确率: TP/(TP+FN)
#mPA/AA 类别平均像素准确率 MPA = (CA1+ ···+CAn)/n
#IoU 交并比 IoU=TP/(TP+TP+FN)
#mIoU 平均交并比 mIoU=(IoU1+···+IoUn)/n

import numpy as np
import copy
from imageio import imsave
from sklearn.metrics import confusion_matrix


sar_colormap = [[141, 211, 199], [255, 255, 179], [190, 185, 218], [251, 128, 114], [128, 177, 211], [179, 222, 105]]
sar_classes = ['bareland', 'woodland', 'water', 'cementfloor', 'road', 'building']


class metric(object):
    def __init__(self, num_class):
        self.numclass = num_class
        self.confusion_matrix = np.zeros((self.numclass,)*2)

    #计算像素准确率PA/OA,即预测正确的像素除以总像素
    def PA(self):
        PA_acc = np.diag(self.confusion_matrix).sum()/(self.confusion_matrix).sum()
        return PA_acc

    #计算OA of per-class/CPA
    def CPA(self):
        #axis按行求和
        CPA_acc = np.diag(self.confusion_matrix)/np.maximum(self.confusion_matrix.sum(axis=1),1)
        return CPA_acc

    #计算AA,类别平均准确率
    def mPA(self):
        CPA_acc = self.CPA()
        #去除背景
        CPA_acc = CPA_acc
        mean_CPA = np.nanmean(CPA_acc)
        return mean_CPA

    #计算mIoU
    def mIoU(self):
        #inter: TP union:TP+FP+FN
        inter = np.diag(self.confusion_matrix)
        union = np.maximum(np.sum(self.confusion_matrix, axis=1)+np.sum(self.confusion_matrix, axis=0)-np.diag(self.confusion_matrix), 1)
        IoU = inter/union
        #去除背景
        IoU = IoU
        mIoU = np.nanmean(IoU)
        return mIoU

    #生成混淆矩阵
    def get_confusion_matrix(self, imgpre, imglabel):
        # print('标签最大', imglabel.max())
        # print('标签最小', imglabel.min())
        # print('预测最大', imgpre.max())
        # print('预测最小', imgpre.min())
        #此处可以选择将背景类去除，不参与计算精度,若移除将类别减一
        mask = (imglabel >= 0) & (imglabel<self.numclass)
        # print(mask)
        label = self.numclass * imglabel[mask].astype(int)+imgpre[mask]
        # print(label)
        # print('label', max(label))
        # print('pre', imgpre)
        # print('label最小', min(label))
        count = np.bincount(label, minlength=self.numclass ** 2)
        confusion_matrix = count.reshape(self.numclass, self.numclass)
        # imglabel_metric = copy.deepcopy(imglabel)
        # imgpre_metric = copy.deepcopy(imgpre)
        # imglabel_metric = imglabel_metric.reshape(-1)
        # imgpre_metric = imgpre_metric.reshape(-1)
        # confusion_all = confusion_matrix(imglabel_metric, imgpre_metric)
        # confusion_de1 = np.delete(confusion_all, 0, 0)
        # confusion_de2 = np.delete(confusion_de1, 0, 1)
        # confusion = confusion_de2

        return confusion_matrix
    
    #去除背景
    def outback(self):
        self.confusion_matrix = np.delete(self.confusion_matrix, 0, 0)
        self.confusion_matrix = np.delete(self.confusion_matrix, 0, 1)


    #更新混淆矩阵
    def addBatch(self, imgpre, imglabel):
        assert imgpre.shape == imglabel.shape
        self.confusion_matrix += self.get_confusion_matrix(imgpre, imglabel)

    #重置混淆矩阵
    def reset(self):
        self.confusion_matrix = np.zeros((self.numclass, self.numclass))


def visual(pred):
    predcolor = copy.deepcopy(pred)
    [row, col] = np.shape(predcolor)
    rgb_color = np.zeros([row, col, 3])
    for i, color in enumerate(sar_colormap):
        colormap_R = np.zeros([row, col])
        colormap_R[(predcolor == i)] = color[0]
        colormap_G = np.zeros([row, col])
        colormap_G[(predcolor == i)] = color[1]
        colormap_B = np.zeros([row, col])
        colormap_B[(predcolor == i)] = color[2]


        color_map = np.stack([colormap_R, colormap_G, colormap_B], axis=0).transpose([1, 2, 0])
        rgb_color = rgb_color+color_map
    visual_map = np.uint8(rgb_color)

    return visual_map

def visual_v2(pred):
    predcolor = copy.deepcopy(pred)
    [row, col] = np.shape(predcolor)
    rgb_color = np.zeros([row, col, 3])
    for i, color in enumerate(sar_colormap):
        colormap_R = np.zeros([row, col])
        colormap_R[(predcolor == i)] = color[0]
        colormap_G = np.zeros([row, col])
        colormap_G[(predcolor == i)] = color[1]
        colormap_B = np.zeros([row, col])
        colormap_B[(predcolor == i)] = color[2]


        color_map = np.stack([colormap_R, colormap_G, colormap_B], axis=0).transpose([1, 2, 0])
        rgb_color = rgb_color+color_map
    #针对不确定像元
    for i in range(predcolor.shape[0]):
        for j in range(predcolor.shape[1]):
            if predcolor[i][j] == 255:
                rgb_color[i,j,0] = 0
                rgb_color[i,j,1] = 0
                rgb_color[i,j,2] = 0
                
    visual_map = np.uint8(rgb_color)

    return visual_map
