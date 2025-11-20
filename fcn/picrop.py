import os
from imageio import imsave
import numpy as np
from PIL import Image
import tifffile as tf

# gt_file = 'D:/AIR-polarsar/Raw_AIR-PolarSAR-Seg/train_set/AIR-PolarSAR-Seg-1_gt.png'
# gt_map = np.array(Image.open(gt_file).convert('RGB'))
# h, w, c = np.shape(gt_map)
# crop_h = int(h/2)
# crop_w = int(w/2)
# gt_map_1 = gt_map[0:crop_h, 0:crop_w, :]
# gt_map_2 = gt_map[0:crop_h:, crop_w:, :]
# gt_map_3 = gt_map[crop_h:, 0:crop_w, :]
# gt_map_4 = gt_map[crop_h:, crop_w:, :]
#
# print(gt_map_1.shape)
# print(gt_map_2.shape)
# print(gt_map_3.shape)
# print(gt_map_4.shape)

root = '/media/disk4T/wr_seg/AIR-polarsar/Raw_AIR-PolarSAR-Seg'
set_name = 'test_set'
save_path = '/media/disk4T/wr_seg/AIR-polarsar/PolarSAR-Seg-crop'
save_root = os.path.join(save_path, set_name)

imageset_name = os.path.join(root, set_name)
image_indexs = []
for file in os.listdir(imageset_name):
    if file.endswith('.png'):
        file_name_gt = file.split('.')
        file_name = file_name_gt[0].split('_')
        image_indexs.append(file_name[0])


for name in image_indexs:
    file_path = os.path.join(root, set_name)
    gt_path = os.path.join(file_path, name+'_gt'+'.png')
    HH_path = os.path.join(file_path, name+'_HH'+'.tiff')
    HV_path = os.path.join(file_path, name + '_HV' + '.tiff')
    VH_path = os.path.join(file_path, name + '_VH' + '.tiff')
    VV_path = os.path.join(file_path, name + '_VV' + '.tiff')

    gt = np.array(Image.open(gt_path).convert('RGB'))
    HH = tf.imread(HH_path)
    HV = tf.imread(HV_path)
    VH = tf.imread(VH_path)
    VV = tf.imread(VV_path)
    #裁块
    h, w, c = np.shape(gt)
    crop_h = int(h/2)
    crop_w = int(w/2)

    #label裁剪
    gt_1 = gt[0:crop_h, 0:crop_w, :]
    gt_2 = gt[0:crop_h:, crop_w:, :]
    gt_3 = gt[crop_h:, 0:crop_w, :]
    gt_4 = gt[crop_h:, crop_w:, :]

    save_gt_1_name = os.path.join(save_root, name+'-1'+'_gt'+'.png')
    imsave(save_gt_1_name, gt_1)
    save_gt_2_name = os.path.join(save_root, name + '-2' + '_gt' + '.png')
    imsave(save_gt_2_name, gt_2)
    save_gt_3_name = os.path.join(save_root, name + '-3' + '_gt' + '.png')
    imsave(save_gt_3_name, gt_3)
    save_gt_4_name = os.path.join(save_root, name + '-4' + '_gt' + '.png')
    imsave(save_gt_4_name, gt_4)

    #HH裁剪
    HH_1 = HH[0:crop_h, 0:crop_w]
    HH_2 = HH[0:crop_h:, crop_w:]
    HH_3 = HH[crop_h:, 0:crop_w]
    HH_4 = HH[crop_h:, crop_w:]

    save_HH_1_name = os.path.join(save_root, name+'-1'+'_HH'+'.npy')
    np.save(save_HH_1_name, HH_1)
    save_HH_2_name = os.path.join(save_root, name + '-2' + '_HH' + '.npy')
    np.save(save_HH_2_name, HH_2)
    save_HH_3_name = os.path.join(save_root, name + '-3' + '_HH' + '.npy')
    np.save(save_HH_3_name, HH_3)
    save_HH_4_name = os.path.join(save_root, name + '-4' + '_HH' + '.npy')
    np.save(save_HH_4_name, HH_4)

    #HV裁剪
    HV_1 = HV[0:crop_h, 0:crop_w]
    HV_2 = HV[0:crop_h:, crop_w:]
    HV_3 = HV[crop_h:, 0:crop_w]
    HV_4 = HV[crop_h:, crop_w:]

    save_HV_1_name = os.path.join(save_root, name+'-1'+'_HV'+'.npy')
    np.save(save_HV_1_name, HV_1)
    save_HV_2_name = os.path.join(save_root, name + '-2' + '_HV' + '.npy')
    np.save(save_HV_2_name, HV_2)
    save_HV_3_name = os.path.join(save_root, name + '-3' + '_HV' + '.npy')
    np.save(save_HV_3_name, HV_3)
    save_HV_4_name = os.path.join(save_root, name + '-4' + '_HV' + '.npy')
    np.save(save_HV_4_name, HV_4)

    #VH裁剪
    VH_1 = VH[0:crop_h, 0:crop_w]
    VH_2 = VH[0:crop_h:, crop_w:]
    VH_3 = VH[crop_h:, 0:crop_w]
    VH_4 = VH[crop_h:, crop_w:]

    save_VH_1_name = os.path.join(save_root, name+'-1'+'_VH'+'.npy')
    np.save(save_VH_1_name, VH_1)
    save_VH_2_name = os.path.join(save_root, name + '-2' + '_VH' + '.npy')
    np.save(save_VH_2_name, VH_2)
    save_VH_3_name = os.path.join(save_root, name + '-3' + '_VH' + '.npy')
    np.save(save_VH_3_name, VH_3)
    save_VH_4_name = os.path.join(save_root, name + '-4' + '_VH' + '.npy')
    np.save(save_VH_4_name, VH_4)

    #VV裁剪
    VV_1 = VV[0:crop_h, 0:crop_w]
    VV_2 = VV[0:crop_h:, crop_w:]
    VV_3 = VV[crop_h:, 0:crop_w]
    VV_4 = VV[crop_h:, crop_w:]

    save_VV_1_name = os.path.join(save_root, name+'-1'+'_VV'+'.npy')
    np.save(save_VV_1_name, VV_1)
    save_VV_2_name = os.path.join(save_root, name + '-2' + '_VV' + '.npy')
    np.save(save_VV_2_name, VV_2)
    save_VV_3_name = os.path.join(save_root, name + '-3' + '_VV' + '.npy')
    np.save(save_VV_3_name, VV_3)
    save_VV_4_name = os.path.join(save_root, name + '-4' + '_VV' + '.npy')
    np.save(save_VV_4_name, VV_4)