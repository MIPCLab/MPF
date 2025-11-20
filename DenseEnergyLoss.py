import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.6")
#sys.path.append("D:\RRM-master\wrapper\\bilateralfilter\\build\lib.linux-x86_64-3.6")
from bilateralfilter import bilateralfilter, bilateralfilter_batch


class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.cuda() #(b, h, w)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1) #(b, K, h, w)全为1矩阵

        seg_max = torch.max(segmentations, dim=1)[0] #返回类别预测概率最大值的值（b, h, w)
        Gate = Gate - seg_max #(b, h, w) 元素为1减去预测的最大值，每个通道都减
        Gate[unlabel_region] = 1 #将没标签（不确定）的区域，Gate设为1，整个过程中就是将预测类别概率小(不确定区域)的地方,gate设为1
        Gate[Gate < 0] = 0 #过小的地方gate设为0, 即预测概率大的地方，Gate为0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1) #将Gate升维度，通道重复，每个通道都有gate

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten() #（b,c,h,w）
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten() #AS对应论文里的G（i,j)
        AS = np.multiply(AS, Gate) #乘以门限，相当于确定出不确定的区域 AS为不确定的区域，确定的区域值为0，对应元素相乘(b,n,h,w).flatten()
        densecrf_loss -= np.dot(segmentations, AS) #添加这个loss的目的是，使得不可信像元与周围像元的亲密度越低越好
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None
    

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images, scale_factor=self.scale_factor) #scale_factor尺寸缩放因子
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest')
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1) #（b,h,w)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )
