from PIL import Image
import numpy as np
from imageio import imsave
import tifffile as tf
import os

# gt_file = 'D:/AIR-polarsar/Raw_AIR-PolarSAR-Seg/train_set/AIR-PolarSAR-Seg-1_gt.png'
# # # # HH = tf.imread(HH_file)
# # # # print(type(HH))
# # # # np.save('HH.npy', HH)
# label_map = np.array(Image.open(gt_file).convert('RGB'))
# # # print(label_map.shape)
# label_map_rot90 = np.rot90(label_map, -3)
# # print(label_map_rot90)
# # if (label_map_rot90 == label_map).all():
# #     print('True')
# # else:
# #     print('False')
# imsave('imagerote2.png', label_map_rot90)

def save_ori(saveroot, name, gt, HH, HV, VH, VV):
    save_gt_name = os.path.join(saveroot, name+'_gt'+'.png')
    imsave(save_gt_name, gt)
    save_HH_name = os.path.join(saveroot, name+'_HH'+'.npy')
    np.save(save_HH_name, HH)
    save_HV_name = os.path.join(saveroot, name + '_HV' + '.npy')
    np.save(save_HV_name, HV)
    save_VH_name = os.path.join(saveroot, name + '_VH' + '.npy')
    np.save(save_VH_name, VH)
    save_VV_name = os.path.join(saveroot, name + '_VV' + '.npy')
    np.save(save_VV_name, VV)

def save_rotate90(saveroot, name, gt, HH, HV, VH, VV):
    save_gt_name = os.path.join(saveroot, name+'-rot90'+'_gt'+'.png')
    gt_rot90 = np.rot90(gt, -1)
    imsave(save_gt_name, gt_rot90)
    save_HH_name = os.path.join(saveroot, name+'-rot90'+'_HH'+'.npy')
    HH_rot90 = np.rot90(HH, -1)
    np.save(save_HH_name, HH_rot90)
    save_HV_name = os.path.join(saveroot, name +'-rot90'+'_HV' + '.npy')
    HV_rot90 = np.rot90(HV, -1)
    np.save(save_HV_name, HV_rot90)
    save_VH_name = os.path.join(saveroot, name + '-rot90'+'_VH' + '.npy')
    VH_rot90 = np.rot90(VH, -1)
    np.save(save_VH_name, VH_rot90)
    save_VV_name = os.path.join(saveroot, name + '-rot90'+'_VV' + '.npy')
    VV_rot90 = np.rot90(VV, -1)
    np.save(save_VV_name, VV_rot90)


def save_rotate180(saveroot, name, gt, HH, HV, VH, VV):
    save_gt_name = os.path.join(saveroot, name+'-rot180'+'_gt'+'.png')
    gt_rot180 = np.rot90(gt, -2)
    imsave(save_gt_name, gt_rot180)
    save_HH_name = os.path.join(saveroot, name+'-rot180'+'_HH'+'.npy')
    HH_rot180 = np.rot90(HH, -2)
    np.save(save_HH_name, HH_rot180)
    save_HV_name = os.path.join(saveroot, name +'-rot180'+'_HV' + '.npy')
    HV_rot180 = np.rot90(HV, -2)
    np.save(save_HV_name, HV_rot180)
    save_VH_name = os.path.join(saveroot, name + '-rot180'+'_VH' + '.npy')
    VH_rot180 = np.rot90(VH, -2)
    np.save(save_VH_name, VH_rot180)
    save_VV_name = os.path.join(saveroot, name + '-rot180'+'_VV' + '.npy')
    VV_rot180 = np.rot90(VV, -2)
    np.save(save_VV_name, VV_rot180)


def save_rotate270(saveroot, name, gt, HH, HV, VH, VV):
    save_gt_name = os.path.join(saveroot, name+'-rot270'+'_gt'+'.png')
    gt_rot270 = np.rot90(gt, -3)
    imsave(save_gt_name, gt_rot270)
    save_HH_name = os.path.join(saveroot, name + '-rot270'+'_HH'+'.npy')
    HH_rot270 = np.rot90(HH, -3)
    np.save(save_HH_name, HH_rot270)
    save_HV_name = os.path.join(saveroot, name +'-rot270'+'_HV' + '.npy')
    HV_rot270 = np.rot90(HV, -3)
    np.save(save_HV_name, HV_rot270)
    save_VH_name = os.path.join(saveroot, name + '-rot270'+'_VH' + '.npy')
    VH_rot270 = np.rot90(VH, -3)
    np.save(save_VH_name, VH_rot270)
    save_VV_name = os.path.join(saveroot, name + '-rot270'+'_VV' + '.npy')
    VV_rot270 = np.rot90(VV, -3)
    np.save(save_VV_name, VV_rot270)



# def save_rotate90(gt, HH, HV, VH, VV):
root = '/media/disk4T/wr_seg/AIR-polarsar/PolarSAR-Seg41'
set_name = 'test_set'
save_path = '/media/disk4T/wr_seg/AIR-polarsar/PolarSAR-Seg41-rot'
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

    #存储
    save_ori(save_root, name, gt, HH, HV, VH, VV)
    # save_rotate90(save_root, name, gt, HH, HV, VH, VV)
    # save_rotate180(save_root, name, gt, HH, HV, VH, VV)
    # save_rotate270(save_root, name, gt, HH, HV, VH, VV)




