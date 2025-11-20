import os
import random
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


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
        name_i_split = name_i.split('_')[-1]
        if name_i_split[0:3] != 'rot':
            file_name_not_rot.append(name_i)
    # print(file_name_not_rot)
    return file_name_not_rot


def read_sar_images(root, is_train=True):
    if is_train:
        set_name = 'train_rot_nozero_set'
    else:
        set_name = 'test_set'
    imageset_name = os.path.join(root, set_name)
    image_indexs = []
    for file in os.listdir(imageset_name):
        if file.endswith(('.tif', '.npy')):
            file_name_gt = file.split('.')
            file_name = file_name_gt[0]
            image_indexs.append(file_name)
    return image_indexs, imageset_name

def get_class_labels(class_root):
    class_labels_dict = np.load(class_root, allow_pickle=True).item()

    return class_labels_dict

def get_zero_class_labels(class_root):
    zero_list = []
    class_labels_dict = np.load(class_root, allow_pickle=True).item()
    for key, value in class_labels_dict.items():
        if (value == [0, 0, 0, 0, 0, 0]).all():
            zero_list.append(key)
    
    return zero_list, class_labels_dict

def normalize1(img):
    mean1 = np.array([39.89])
    std1 = np.array([27.549])
    norm_img = (img-mean1)/std1

    return norm_img

class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):
        #indices_from表示源切片 indices_from表示目标切片
        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map, segm_map1):

        segm_map_flat = np.reshape(segm_map, -1)
        segm_map_flat_1 = np.reshape(segm_map1, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]
        segm_label_from_1 = np.expand_dims(segm_map_flat_1[self.indices_from], axis=0)
        segm_label_to_1 = segm_map_flat_1[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 6), np.less(segm_label_to, 6))
        valid_label_1 = np.logical_and(np.less(segm_label_from_1, 6), np.less(segm_label_to_1, 6))
        valid_label_all = np.logical_and(valid_label, valid_label_1)

        equal_label = np.equal(segm_label_from, segm_label_to)
        equal_label_1 = np.equal(segm_label_from_1, segm_label_to_1)
        equal_label_all = np.logical_and(equal_label, equal_label_1)

        pos_affinity_label = np.logical_and(equal_label, valid_label)
        pos_affinity_label_1 = np.logical_and(equal_label_1, valid_label_1)
        pos_affinity_label_all = np.logical_and(pos_affinity_label, pos_affinity_label_1)
        

        # bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label_all, np.greater(segm_label_from, -1)).astype(np.float32)
        # print(fg_pos_affinity_label.shape)

        # neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label_all).astype(np.float32)
        neg_affinity_label = np.logical_and(np.logical_not(equal_label_all), valid_label_all).astype(np.float32)


        return torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)
    

class sarsegDataset_irn(Dataset):
    def __init__(self, sar_root, irn_layer1_label_root, irn_layer2_label_root, crop_size, indices_from, indices_to, is_train, rescale=None):
        super().__init__()

        self.sar_root = sar_root
        self.image_name_indexs_all, self.imageset_name = read_sar_images(sar_root, is_train=is_train)
        self.image_name_indexs = self.image_name_indexs_all
        print('语义分割网络训练样本', len(self.image_name_indexs))
        # print(len(self.image_name_indexs))
        self.is_train = is_train
        self.irn_layer1_label = irn_layer1_label_root
        self.irn_layer2_label = irn_layer2_label_root
        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)
        # f = open("traintest.txt","a")
        # f.write(self.imageset_name+'\n')
        # f.close()

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, item):
        image_name = self.image_name_indexs[item]
        print(image_name)
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
        #sar特征
        sar_feature = np.load(os.path.join(self.imageset_name, image_name+'.npy'))
        sar_feature = normalize1(sar_feature)
        sar_feature = sar_feature.astype(np.float32)
        sar_feature = np.stack([sar_feature, sar_feature, sar_feature], axis=0)

        # #span归一化
        # file_HH = os.path.join(self.imageset_name, image_name+'_HH'+'.npy')
        # data_HH = np.load(file_HH).astype(float)
        # file_HV = os.path.join(self.imageset_name, image_name+'_HV'+'.npy')
        # data_HV = np.load(file_HV).astype(float)
        # file_VH = os.path.join(self.imageset_name, image_name+'_VH'+'.npy')
        # data_VH = np.load(file_VH).astype(float)
        # file_VV = os.path.join(self.imageset_name, image_name+'_VV'+'.npy')
        # data_VV = np.load(file_VV).astype(float)
        # # #计算总功率span
        # span = data_HH**2 + data_HV**2 + data_VH**2 + data_VV**2
        # # span = normalize(span)
        # data_HH = np.divide(data_HH, span, np.zeros_like(data_HH), where=span!=0)
        # data_HV = np.divide(data_HV, span, np.zeros_like(data_HV), where=span!=0)
        # data_VH = np.divide(data_VH, span, np.zeros_like(data_VH), where=span!=0)
        # data_VV = np.divide(data_VV, span, np.zeros_like(data_VV), where=span!=0)
        # #合并极化通道
        # #HH_VV = np.divide(data_HH, data_VV, np.zeros_like(data_HH), where=data_VV!=0)
        # sar_feature = np.stack([data_HH, (data_HV+data_VH)*0.5, data_VV], axis=0)
        # sar_feature = np.stack([data_HH, data_HV, data_VH, data_VV, span], axis=0)
        #读取irn标签
        #读取伪标签
        irn_layer1_label_path = os.path.join(self.irn_layer1_label, image_name+'.npy')
        irn_layer2_label_path = os.path.join(self.irn_layer2_label, image_name+'.npy')
        # label_map = Image.open(file_gt).convert('RGB')
        # label = rgbtruth_to_label(label_map, colormap_to_label)
        irn_layer1_label = np.load(irn_layer1_label_path)
        irn_layer2_label = np.load(irn_layer2_label_path)
        # irn_label = torch.from_numpy(irn_label)
        sar_feature = sar_feature.astype(np.float32)
        # print('形状', sar_feature.shape)
        # sar_feature = torch.FloatTensor(sar_feature)
        # irn_label = torch.from_numpy(irn_label)
        # irn_label = irn_label.long()
        aff_fg_pos_label, aff_neg_label = self.extract_aff_lab_func(irn_layer1_label, irn_layer2_label)

        return image_name, sar_feature, irn_layer1_label, irn_layer2_label, aff_fg_pos_label, aff_neg_label

    def __len__(self):
        return len(self.image_name_indexs)



class sarsegDataset_infer(Dataset):
    def __init__(self, sar_root, is_train):
        super().__init__()

        self.sar_root = sar_root
        self.image_name_indexs_all, self.imageset_name = read_sar_images(sar_root, is_train=is_train)
        self.image_name_indexs = remove_rot(self.image_name_indexs_all)
        print('语义分割网络训练样本', len(self.image_name_indexs))
        # print(len(self.image_name_indexs))
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
        sar_feature = np.load(os.path.join(self.imageset_name, image_name+'.npy'))
        sar_feature = normalize1(sar_feature)
        sar_feature = sar_feature.astype(np.float32)
        sar_feature = np.stack([sar_feature, sar_feature, sar_feature], axis=0)
        sar_feature = torch.FloatTensor(sar_feature)

        # #span归一化
        # file_HH = os.path.join(self.imageset_name, image_name+'_HH'+'.npy')
        # data_HH = np.load(file_HH).astype(float)
        # file_HV = os.path.join(self.imageset_name, image_name+'_HV'+'.npy')
        # data_HV = np.load(file_HV).astype(float)
        # file_VH = os.path.join(self.imageset_name, image_name+'_VH'+'.npy')
        # data_VH = np.load(file_VH).astype(float)
        # file_VV = os.path.join(self.imageset_name, image_name+'_VV'+'.npy')
        # data_VV = np.load(file_VV).astype(float)
        # # #计算总功率span
        # span = data_HH**2 + data_HV**2 + data_VH**2 + data_VV**2
        # # span = normalize(span)
        # data_HH = np.divide(data_HH, span, np.zeros_like(data_HH), where=span!=0)
        # data_HV = np.divide(data_HV, span, np.zeros_like(data_HV), where=span!=0)
        # data_VH = np.divide(data_VH, span, np.zeros_like(data_VH), where=span!=0)
        # data_VV = np.divide(data_VV, span, np.zeros_like(data_VV), where=span!=0)
        # #合并极化通道
        # #HH_VV = np.divide(data_HH, data_VV, np.zeros_like(data_HH), where=data_VV!=0)
        # sar_feature = np.stack([data_HH, (data_HV+data_VH)*0.5, data_VV], axis=0)
        # sar_feature = sar_feature.astype(np.float32)
        # sar_feature = torch.FloatTensor(sar_feature)
        image_name_gt = image_name

        return image_name_gt, sar_feature

    def __len__(self):
        return len(self.image_name_indexs)
