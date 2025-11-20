import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES']='4'

import skimage
# from misc import torchutils, imutils, pyutils, indexing
from misc import indexing
from net import resnet50_irn
from dataloader_irn import sarsegDataset_infer
from argparse import ArgumentParser
import random
from segmetric import metric, visual, visual_v2
from imageio import imsave


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description='SAR RRM')
parser.add_argument('--dataroot', type=str, default='/media/disk8T/more2/wr/eeds/sar_data', help='Dataset root')
parser.add_argument('--irn_label_root', type=str, default='/media/disk8T/more2/wr/RRM_eeds/cam/label', help='Dataset root')
parser.add_argument('--labelroot', type=str, default='/media/disk8T/more2/wr/eeds/label_data/label_class.npy', help='Dataset root')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--workers', type=int, default=1, help='Data loader workers')
parser.add_argument('--crop_size', type=int, default=256, help='Img size')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_cls', type=int, default=6, help='number of classification classes')
parser.add_argument('--num_seg', type=int, default=6, help='number of segmentation classes')
parser.add_argument("--irn_num_epoches", default=6, type=int)
parser.add_argument("--irn_lr", default=0.01, type=float)
parser.add_argument("--irn_wt_dec", default=5e-4, type=float)
parser.add_argument("--beta", default=5)
parser.add_argument("--exp_times", default=7,help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
parser.add_argument("--session_name", default="SAR-RRM", type=str)
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = sarsegDataset_infer(args.dataroot, is_train=True)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

model = resnet50_irn.EdgeDisplacement()
model.load_state_dict(torch.load('/media/disk8T/more2/wr/RRM_eeds/RRM/model/irn.pth'), strict=False)
model.to(device)
model.eval()

with torch.no_grad():
    for iter, pack in enumerate(data_loader):
        image_name, sar_feature = pack
        sar_feature = sar_feature.to(device)
        orig_img_size = 256
        
        edge, dp = model(sar_feature)
        layer1_cam_dict = np.load('/media/disk8T/more2/wr/RRM_eeds/cam/dict_label/' + image_name[0] + '.npy', allow_pickle=True).item()
        layer2_cam_dict = np.load('/media/disk8T/more2/wr/RRM_eeds/layer_cam/layer3/cam/dict_pred/' + image_name[0] + '.npy', allow_pickle=True).item()
        layer3_cam_dict = np.load('/media/disk8T/more2/wr/RRM_eeds/layer_cam/layer2/cam/dict_pred/' + image_name[0] + '.npy', allow_pickle=True).item()
        layer4_cam_dict = np.load('/media/disk8T/more2/wr/RRM_eeds/layer_cam/layer1/cam/dict_pred/' + image_name[0] + '.npy', allow_pickle=True).item()

        cams_layer1 = layer1_cam_dict['cam']
        keys_layer1 = layer1_cam_dict['keys']

        cams_layer2 = layer2_cam_dict['cam']
        keys_layer2 = layer2_cam_dict['keys']

        cams_layer3 = layer3_cam_dict['cam']
        keys_layer3 = layer3_cam_dict['keys']

        cams_layer4 = layer4_cam_dict['cam']
        keys_layer4 = layer4_cam_dict['keys']
        # print(cams_layer2)

        layer1_cam_values = cams_layer1.to(device)
        layer2_cam_values = torch.from_numpy(cams_layer2).to(torch.float32)
        layer2_cam_values = layer2_cam_values.to(device)
        layer3_cam_values = torch.from_numpy(cams_layer3).to(torch.float32)
        layer3_cam_values = layer3_cam_values.to(device)
        layer4_cam_values = torch.from_numpy(cams_layer4).to(torch.float32)
        layer4_cam_values = layer4_cam_values.to(device)

        # cam_values = layer1_cam_values
        # cam_values = layer4_cam_values
        cam_values = layer1_cam_values + layer3_cam_values*0.5       # print(keys)
        # cam_values = (layer2_cam_values + layer3_cam_values)*0.5 + layer1_cam_values
        # cam_values = (layer2_cam_values + layer3_cam_values + layer4_cam_values) *0.2 + layer1_cam_values
        # cam_values /= F.adaptive_max_pool2d(cam_values, (1, 1)) + 1e-5 
        # cam_values = layer1_cam_values
        # print(torch.argmax(cam_values, dim=0))
        # # print(cam_values.shape)
        # print(keys)

        rw = indexing.propagate_to_edge(cam_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
        # print(rw)
        rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :256, :256]
        # print(rw_up)
        rw_up = rw_up / (torch.max(rw_up) + 1e-5)
        # print(rw_up)
        rw_pred = torch.argmax(rw_up, dim=0).cpu().numpy()
        #print(rw_pred)
        keys = keys_layer1.cpu().numpy()
        # print(rw_pred)
        # print(rw_pred.shape)

        rw_pred = keys[rw_pred]
        # print(rw_pred)
        
        np.save('/media/disk8T/more2/wr/RRM_eeds/RRM/label/'+ image_name[0] + '.npy', rw_pred.astype(np.uint8))
        seg_label_map = visual_v2(rw_pred)
        imsave('/media/disk8T/more2/wr/RRM_eeds/RRM/visual_label/'+ image_name[0] + '.png', seg_label_map)

        


