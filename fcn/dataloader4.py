import os
import random
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
# import gdal
import tifffile as tf


# sar_colormap = [[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [255, 255, 255], [255, 255, 0]]
# sar_classes = ['background', 'industry', 'natural', 'landuse', 'water', 'other', 'housing']

sar_colormap = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255], [255, 255, 255], [255, 255, 0]]
sar_classes = ['industry', 'natural', 'landuse', 'water', 'other', 'housing']

colormap_to_label = torch.zeros(256**3, dtype=torch.uint8)
for i, colormap in enumerate(sar_colormap):
    colormap_to_label[(colormap[0]*256 + colormap[1])*256 + colormap[2]] = i

def rgbtruth_to_label(colormap, colormap_to_label):
    colormap = np.array(colormap.convert('RGB')).astype('int32')
    idx = ((colormap[:, :, 0]*256+colormap[:, :, 1])*256+colormap[:, :, 2])

    return colormap_to_label[idx]

def remove_rot(image_indexs):
    file_name_not_rot = []
    for name_i in image_indexs:
        name_i_split = name_i.split('-')[-2]
        if name_i_split[0:3] != 'rot':
            file_name_not_rot.append(name_i)
    return file_name_not_rot

def read_sar_images(root, is_train=True):
    if is_train:
        set_name = 'train_set'
    else:
        set_name = 'test_set'
    imageset_name = os.path.join(root, set_name)
    image_indexs = []
    for file in os.listdir(imageset_name):
        if file.endswith('.png'):
            file_name_gt = file.split('.')
            file_name = file_name_gt[0].split('_')
            image_indexs.append(file_name[0])
    final_indexs = remove_rot(image_indexs)
    return final_indexs, imageset_name

def normalize(img):
    h, w = np.shape(img)
    img = img.reshape(-1)
    if np.max(img) == 0:
        img = img
    else:
        img = (img - np.min(img))/(np.max(img) - np.min(img))
    img = img.reshape([h, w])

    return img


class sarsegDataset_train(Dataset):
    def __init__(self, sar_root, pseudo_root, is_train):
        super().__init__()

        self.sar_root = sar_root
        self.image_name_indexs, self.imageset_name = read_sar_images(sar_root, is_train=is_train)
        print(len(self.image_name_indexs))
        self.is_train = is_train
        self.pseduo_label = pseudo_root
        # f = open("traintest.txt","a")
        # f.write(self.imageset_name+'\n')
        # f.close()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, item):
        image_name = self.image_name_indexs[item]
        if self.is_train == False:
            f = open("traintest.txt","a")
            f.write(self.imageset_name+'\n')
            f.write(image_name+'\n')
        # print('图片名称', image_name)
        #读取SAR不同极化通道数据
        # file_HH = os.path.join(self.imageset_name, image_name+'_HH'+'.npy')
        # # data_HH = gdal.Open(file_HH).ReadAsArray()
        # data_HH = normalize(np.load(file_HH))
        # file_HV = os.path.join(self.imageset_name, image_name+'_HV'+'.npy')
        # # data_HV = gdal.Open(file_HV).ReadAsArray()
        # data_HV = normalize(np.load(file_HV))
        # file_VH = os.path.join(self.imageset_name, image_name+'_VH'+'.npy')
        # # data_VH = gdal.Open(file_VH).ReadAsArray()
        # data_VH = normalize(np.load(file_VH))
        # file_VV = os.path.join(self.imageset_name, image_name+'_VV'+'.npy')
        # # data_VV = gdal.Open(file_VV).ReadAsArray()
        # data_VV = normalize(np.load(file_VV))

        #span归一化
        file_HH = os.path.join(self.imageset_name, image_name+'_HH'+'.npy')
        data_HH = np.load(file_HH).astype(float)
        file_HV = os.path.join(self.imageset_name, image_name+'_HV'+'.npy')
        data_HV = np.load(file_HV).astype(float)
        file_VH = os.path.join(self.imageset_name, image_name+'_VH'+'.npy')
        data_VH = np.load(file_VH).astype(float)
        file_VV = os.path.join(self.imageset_name, image_name+'_VV'+'.npy')
        data_VV = np.load(file_VV).astype(float)
        # #计算总功率span
        span = data_HH**2 + data_HV**2 + data_VH**2 + data_VV**2
        # span = normalize(span)
        data_HH = np.divide(data_HH, span, np.zeros_like(data_HH), where=span!=0)
        data_HV = np.divide(data_HV, span, np.zeros_like(data_HV), where=span!=0)
        data_VH = np.divide(data_VH, span, np.zeros_like(data_VH), where=span!=0)
        data_VV = np.divide(data_VV, span, np.zeros_like(data_VV), where=span!=0)
        #合并极化通道
        #HH_VV = np.divide(data_HH, data_VV, np.zeros_like(data_HH), where=data_VV!=0)
        sar_feature = np.stack([data_HH, (data_HV+data_VH)*0.5, data_VV], axis=0)
        # sar_feature = np.stack([data_HH, data_HV, data_VH, data_VV, span], axis=0)
        #读取伪标签
        pseudo_gt = os.path.join(self.pseduo_label, image_name+'_gt'+'.npy')
        # label_map = Image.open(file_gt).convert('RGB')
        # label = rgbtruth_to_label(label_map, colormap_to_label)
        pseudo_gt = np.load(pseudo_gt)
        pseudo_gt = torch.from_numpy(pseudo_gt)
        sar_feature = sar_feature.astype(np.float32)
        # print('形状', sar_feature.shape)
        sar_feature = torch.FloatTensor(sar_feature)
        # label = torch.from_numpy(label.values)
        pseudo_gt = pseudo_gt.long()

        return sar_feature, pseudo_gt

    def __len__(self):
        return len(self.image_name_indexs)


class sarsegDataset_test(Dataset):
    def __init__(self, sar_root, is_train):
        super().__init__()

        self.sar_root = sar_root
        self.image_name_indexs, self.imageset_name = read_sar_images(sar_root, is_train=is_train)
        print(len(self.image_name_indexs))
        self.is_train = is_train
        # f = open("traintest.txt","a")
        # f.write(self.imageset_name+'\n')
        # f.close()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, item):
        image_name = self.image_name_indexs[item]
        if self.is_train == False:
            f = open("traintest.txt","a")
            f.write(self.imageset_name+'\n')
            f.write(image_name+'\n')
        # print('图片名称', image_name)
        #读取SAR不同极化通道数据
        # file_HH = os.path.join(self.imageset_name, image_name+'_HH'+'.npy')
        # # data_HH = gdal.Open(file_HH).ReadAsArray()
        # data_HH = normalize(np.load(file_HH))
        # file_HV = os.path.join(self.imageset_name, image_name+'_HV'+'.npy')
        # # data_HV = gdal.Open(file_HV).ReadAsArray()
        # data_HV = normalize(np.load(file_HV))
        # file_VH = os.path.join(self.imageset_name, image_name+'_VH'+'.npy')
        # # data_VH = gdal.Open(file_VH).ReadAsArray()
        # data_VH = normalize(np.load(file_VH))
        # file_VV = os.path.join(self.imageset_name, image_name+'_VV'+'.npy')
        # # data_VV = gdal.Open(file_VV).ReadAsArray()
        # data_VV = normalize(np.load(file_VV))

        #span归一化
        file_HH = os.path.join(self.imageset_name, image_name+'_HH'+'.npy')
        data_HH = np.load(file_HH).astype(float)
        file_HV = os.path.join(self.imageset_name, image_name+'_HV'+'.npy')
        data_HV = np.load(file_HV).astype(float)
        file_VH = os.path.join(self.imageset_name, image_name+'_VH'+'.npy')
        data_VH = np.load(file_VH).astype(float)
        file_VV = os.path.join(self.imageset_name, image_name+'_VV'+'.npy')
        data_VV = np.load(file_VV).astype(float)
        # #计算总功率span
        span = data_HH**2 + data_HV**2 + data_VH**2 + data_VV**2
        # span = normalize(span)
        data_HH = np.divide(data_HH, span, np.zeros_like(data_HH), where=span!=0)
        data_HV = np.divide(data_HV, span, np.zeros_like(data_HV), where=span!=0)
        data_VH = np.divide(data_VH, span, np.zeros_like(data_VH), where=span!=0)
        data_VV = np.divide(data_VV, span, np.zeros_like(data_VV), where=span!=0)
        #合并极化通道
        #HH_VV = np.divide(data_HH, data_VV, np.zeros_like(data_HH), where=data_VV!=0)
        sar_feature = np.stack([data_HH, (data_HV+data_VH)*0.5, data_VV], axis=0)
        # sar_feature = np.stack([data_HH, data_HV, data_VH, data_VV, span], axis=0)
        #读取标签RGB，转为数字标签
        file_gt = os.path.join(self.imageset_name, image_name+'_gt'+'.png')
        label_map = Image.open(file_gt).convert('RGB')
        label = rgbtruth_to_label(label_map, colormap_to_label)
        sar_feature = sar_feature.astype(np.float32)
        # print('形状', sar_feature.shape)
        sar_feature = torch.FloatTensor(sar_feature)
        # label = torch.from_numpy(label.values)
        label = label.long()

        return image_name, sar_feature, label

    def __len__(self):
        return len(self.image_name_indexs)
    
# sar_root = '/media/disk4T/wr_seg/AIR-polarsar/Raw_AIR-PolarSAR-Seg'
# sar_train = sarsegDataset(sar_root, False)
# print(len(sar_train))