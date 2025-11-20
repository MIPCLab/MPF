# pip install importlib_resources

import torch
import torch.nn.functional as F
import torchvision.models as models
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from dataloader import sarsegDataset_train
from argparse import ArgumentParser

from utils import *
from cam_sar.layercam import *
from model_rec import Net, FeatureResNet, CAM
import random
from segmetric import metric, visual, visual_v2
from imageio import imsave
from tool.myTool import compute_seg_label, compute_joint_loss, compute_cam_up, compute_joint_loss_SAR, compute_seg_label_v2, compute_seg_label_v3


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = ArgumentParser(description='SAR RRM')
parser.add_argument('--dataroot', type=str, default='/media/disk8T/wr/PolSAR-Seg41-rot-crop-movezero', help='Dataset root')
parser.add_argument('--labelroot', type=str, default='/media/disk8T/wr/PolSAR-Seg41-rot-crop-movezero/label_class.npy', help='Dataset root')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--workers', type=int, default=1, help='Data loader workers')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--num_cls', type=int, default=6, help='number of classification classes')
parser.add_argument('--num_seg', type=int, default=6, help='number of segmentation classes')
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)


def get_data(image_name):
    input_path = '/media/disk8T/wr/PolSAR-Seg41-rot-crop-movezero/train_set'
    data_HH_path = os.path.join(input_path, image_name+'_HH'+'.npy')
    data_HH = np.load(data_HH_path).astype(float)
    data_HV_path = os.path.join(input_path, image_name+'_HV'+'.npy')
    data_HV = np.load(data_HV_path).astype(float)
    data_VH_path = os.path.join(input_path, image_name+'_VH'+'.npy')
    data_VH = np.load(data_VH_path).astype(float) 
    data_VV_path = os.path.join(input_path, image_name+'_VV'+'.npy')
    data_VV = np.load(data_VV_path).astype(float)

    #归一化数据处理
    span = data_HH**2 + data_HV**2 + data_VH**2 + data_VV**2
    # span = normalize(span)
    data_HH = np.divide(data_HH, span, np.zeros_like(data_HH), where=span!=0)
    data_HV = np.divide(data_HV, span, np.zeros_like(data_HV), where=span!=0)
    data_VH = np.divide(data_VH, span, np.zeros_like(data_VH), where=span!=0)
    data_VV = np.divide(data_VV, span, np.zeros_like(data_VV), where=span!=0)
    #合并极化通道
    #HH_VV = np.divide(data_HH, data_VV, np.zeros_like(data_HH), where=data_VV!=0)
    sar_feature = np.stack([data_HH, (data_HV+data_VH)*0.5, data_VV], axis=0)

    sar_feature = sar_feature.astype(np.float32)
    # print('形状', sar_feature.shape)
    sar_feature = torch.FloatTensor(sar_feature).unsqueeze(0)

    return sar_feature


def vote_max(final, layer2, layer3, w, h):
    seg_label = np.zeros([w, h])
    pred_all = np.stack([final, layer2, layer3], axis=0)
    for i in range(pred_all.shape[1]):
        for j in range(pred_all.shape[2]):
            pred_idx = pred_all[:, i, j]
            pred_idx_count = np.bincount(pred_idx)
            pred_max = np.argmax(pred_idx_count)
            seg_label[i, j] = pred_max
    
    return seg_label

def compute_seg_label_layer(ori_img, cam_label, norm_cam):
    cam_label = cam_label.astype(np.uint8)
    cam_all = norm_cam
    
    # cam_np = np.zeros_like(norm_cam)
    # for i in range(6):
    #     if cam_label[i] > 1e-5:
    #         cam_np[i] = norm_cam[i]

    cam_img = np.argmax(cam_all, 0)
    cam_label = cam_img.copy()
    # cam_seg_label_true = cam_label[cam_seg_label]

    single_img_classes = np.unique(cam_img)
    cam_sure_region = np.zeros([256, 256], dtype=bool)
    for class_i in single_img_classes:
        class_not_region = (cam_img != class_i)
        cam_class = cam_all[class_i, :, :]
        cam_class[class_not_region] = 0
        cutoff_ori = 0.05
        if cam_class.max() < cutoff_ori:
            cutoff = cam_class.min()
        else:
            cutoff = cutoff_ori
    
        cam_class_order = cam_class[cam_class >= cutoff]
        cam_class_order = np.sort(cam_class_order)
        confidence_pos = int(cam_class_order.shape[0] * 0.2)
        confidence_value = cam_class_order[confidence_pos]
        class_sure_region = (cam_class > confidence_value)
        cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
    
    cam_not_sure_region = ~cam_sure_region
    not_sure_region = cam_not_sure_region
    cam_label[not_sure_region] = 255

    return cam_label




pseudo_dataset = sarsegDataset_train(sar_root=args.dataroot, class_root=args.labelroot, is_train=True, re_rot=True)
pseudo_loader = DataLoader(pseudo_dataset, batch_size=1,shuffle=True, num_workers=args.workers)
    

pretrained_net = FeatureResNet()
model_layer = Net(6, pretrained_net).to(device)
model_layer.load_state_dict(torch.load('/media/disk8T/wr/SAR-RRM/cam-10.pth'))
model_layer.eval()

layer_name_3 = 'layer3'
# print(model)
model_dict_3 = dict(arch=model_layer, layer_name=layer_name_3, input_size=(256,256))
model_layercam_layer3 = LayerCAM(model_dict_3)

layer_name_2 = 'layer2'
model_dict_2 = dict(arch=model_layer, layer_name=layer_name_2, input_size=(256,256))
model_layercam_layer2 = LayerCAM(model_dict_2)

#最后一层cam
pretrain_net = FeatureResNet()
model_cam = CAM(args.num_cls, pretrain_net)
model_cam.to(device)
model_cam.load_state_dict(torch.load('/media/disk8T/wr/SAR-RRM/cam-10.pth'))
model_cam.eval()

#获取不同层的预测
for i, pack in enumerate(pseudo_loader):
    img_name = pack[0]
    ori_image = pack[2].numpy()
    image_fea = pack[1].to(device)
    class_label = pack[3].to(device)

    b, _, w, h = image_fea.shape
    class_num = args.num_cls

    predicted_class = model_layer(image_fea).max(1)[-1]
    class_idx = class_label.cpu().numpy()
    class_idx = class_idx.reshape([6])
    # print(class_idx.shape)
    class_idx_diag = np.diag(class_idx)
    class_idx_indexs = class_idx_diag

    # class_idx_indexs = class_idx_diag[~np.all(class_idx_diag == 0, axis=1)]
    # class_name = np.where(class_idx == 1)[0]

    #计算layer_cam layer3
    cam_all_layer3 = []
    cam_all_layer2 = []
    zero_map = np.zeros((256, 256))
    for i in range(class_idx_indexs.shape[0]):
        class_idx_i = class_idx_indexs[i]
        if np.all(class_idx_i == 0):
            cam_all_layer3.append(zero_map)
            cam_all_layer2.append(zero_map)
        else:
            layercam_map_layer3 = model_layercam_layer3(image_fea, class_idx=class_idx_i)
            layercam_map_layer3 = layercam_map_layer3.cpu().numpy().reshape([256, 256])
            cam_all_layer3.append(layercam_map_layer3)

            layercam_map_layer2 = model_layercam_layer2(image_fea, class_idx=class_idx_i)
            layercam_map_layer2 = layercam_map_layer2.cpu().numpy().reshape([256, 256])
            cam_all_layer2.append(layercam_map_layer2)
    
    cam_norm_layer3 = np.stack([i for i in cam_all_layer3], axis=0)
    cam_norm_layer2 = np.stack([i for i in cam_all_layer2], axis=0)

    #计算最后一层cam
    final_out = model_cam(image_fea)
    final_cam = final_out
    final_cam_up = compute_cam_up(final_cam, class_label, w, h, b) #输出为numpy
    final_cam_up = np.reshape(final_cam_up, [6, w, h])

    # cam_fuse = (layer_cam_norm + final_cam_up) * 0.5

    cam_norm_final = final_cam_up / (np.max(final_cam_up, (1, 2), keepdims=True) + 1e-5)

    seg_label_layer2 = compute_seg_label_v3(ori_image, class_idx, cam_norm_layer2) 
    seg_label_layer3 = compute_seg_label_v3(ori_image, class_idx, cam_norm_layer3) 
    seg_label_final = compute_seg_label_v3(ori_image, class_idx, cam_norm_final)

    seg_label = vote_max(seg_label_final, seg_label_layer2, seg_label_layer3, w, h)

    #seg_label = compute_seg_label_layer(ori_image, class_idx, cam_mat_all_norm)
    np.save('/media/disk8T/wr/pseudo_layer/'+ img_name[0] + '.npy', seg_label)
    seg_label_map = visual_v2(seg_label)
    imsave('/media/disk8T/wr/pseudo_layer/'+ img_name[0] + '.png', seg_label_map)

print('完成')