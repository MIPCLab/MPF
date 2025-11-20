import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
import os
import random
from dataloader import sarsegDataset_train, sarsegDataset_test
from torch.utils.data import DataLoader
# from model import SegNet, FeatureResNet
from model_rec import Net_CAM_Feature, FeatureResNet, Class_Predictor, CAM, Net, Net_CAM
from torchvision import models
from tool import pyutils, imutils, torchutils
from tool.myTool import compute_seg_label, compute_joint_loss, compute_cam_up, compute_cam_up_v2, compute_joint_loss_SAR, compute_seg_label_v2, compute_seg_label_v3
from segmetric import metric, visual, visual_v2
from imageio import imsave
import copy


os.environ['TORCH_HOME'] = '/media/disk8T/wr/torch-model'


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description='SAR RRM')
parser.add_argument('--dataroot', type=str, default='/media/disk8T/wr/PolSAR-Seg41-rot-crop-movezero', help='Dataset root')
parser.add_argument('--labelroot', type=str, default='/media/disk8T/wr/PolSAR-Seg41-rot-crop-movezero/label_class.npy', help='Dataset root')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--workers', type=int, default=1, help='Data loader workers')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--num_cls', type=int, default=6, help='number of classification classes')
parser.add_argument('--num_seg', type=int, default=6, help='number of segmentation classes')
parser.add_argument("--max_epoches", default=10, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--wt_dec", default=5e-4, type=float)
parser.add_argument("--session_name", default="SAR-RRM", type=str)
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#dataloader
train_dataset = sarsegDataset_train(sar_root=args.dataroot, class_root=args.labelroot, is_train=True, re_rot=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.workers)
test_dataset = sarsegDataset_test(sar_root=args.dataroot, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
pseudo_dataset = sarsegDataset_train(sar_root=args.dataroot, class_root=args.labelroot, is_train=True, re_rot=False)
pseudo_loader = DataLoader(pseudo_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

# print('长度', train_dataset.__len__())

# save_path = os.path.join("/media/disk8T/wr", args.session_name)
save_path = '/media/disk8T/more2/wr/RRM/cam_model'
# print('保存路径', save_path)
#定义网络
#backbone ResNet34
pretrained_net = FeatureResNet()
pretrained_net.load_state_dict(models.resnet50(pretrained=True).state_dict())
model = Net(args.num_cls, pretrained_net)

# recam_predictor = Class_Predictor(args.num_cls, 2048)
# recam_predictor.to(device)
# recam_predictor.train()

critersion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='elementwise_mean')

pyutils.Logger(args.session_name + '.log')

max_step = (train_dataset.__len__() // args.batch_size) * args.max_epoches
# max_step = 10

param_groups = model.get_parameter_groups()
optimizer = torchutils.PolyOptimizer([
    {'params': param_groups[0], 'lr':args.lr, 'weight_decay': args.wt_dec},
    {'params': param_groups[1], 'lr':2*args.lr, 'weight_decay': 0},
    {'params': param_groups[2], 'lr':10*args.lr, 'weight_decay': args.wt_dec},
    {'params': param_groups[3], 'lr':20*args.lr, 'weight_decay': 0},
], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

# model = model.to(device)
# model.train()

# avg_meter = pyutils.AverageMeter('loss')
# timer = pyutils.Timer('Session started:')

# for epoch in range(args.max_epoches):
#     for i, pack in enumerate(train_loader):
#         img_name = pack[0]
#         ori_image = pack[2].numpy() #输入原始未归一化数据（b,c,h,w）
#         image_fea = pack[1].to(device) #输入图像数据
#         class_label = pack[3].to(device) #one-hot标签
#         # print(img_name)

#         b, _, w, h = image_fea.shape
#         class_num = args.num_cls

#         x = model(image_fea)

#         closs = F.multilabel_soft_margin_loss(x, class_label)

#         # x_f = model(image_fea, require_seg=False, require_mcam=False)
#         # closs = F.multilabel_soft_margin_loss(x_f, class_label)
#         loss = closs
#         print('closs', loss.data)
#         avg_meter.add({'loss': loss.item()})
#         # avg_meter.add({'loss_ce': loss_ce.item()})
#         # avg_meter.add({'acc': acc.item()})

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (optimizer.global_step-1)%50 == 0:
#             timer.update_progress(optimizer.global_step / max_step)

#             print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
#                   'Loss:%.4f' % (avg_meter.pop('loss')),
#                   'imps:%.1f' % ((i + 1) * args.batch_size / timer.get_stage_elapsed()),
#                   'Fin:%s' % (timer.str_est_finish()),
#                   'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

#             # if (optimizer.global_step - 1) % 10000 == 0 and optimizer.global_step > 10000:
#             #     torch.save(model.state_dict(), save_path + '%d.pth' % (optimizer.global_step - 1))

#     else:
#         timer.reset_stage()

# torch.save(model.state_dict(), os.path.join(save_path, 'cam-10.pth'))
# # torch.save(recam_predictor.state_dict(), save_path + args.session_name + 'recam_predictor.pth')
# print('已存入cam模型')



#获取类激活映射
#获取伪标签
model_cam = CAM(args.num_cls, pretrained_net)
model_cam.to(device)
model_cam.load_state_dict(torch.load('/media/disk8T/more2/wr/RRM/cam_model/cam-10.pth'))
# recam_predictor.load_state_dict(torch.load('/media/disk8T/wr/SAR-RRM/recam_predictor.pth'))
model_cam.eval()
# recam_predictor.eval()
with torch.no_grad():
    for i, pack in enumerate(pseudo_loader):
        img_name = pack[0]
        ori_image = pack[2].numpy() #输入原始未归一化数据（b,c,h,w）
        image_fea = pack[1].to(device) #输入图像数据
        class_label = pack[3].to(device) #one-hot标签

        b, _, w, h = image_fea.shape
        class_num = args.num_cls

        outputs = model_cam(image_fea)
        cam = outputs
        # print(cam)
        # print('cam形状', cam.shape)

        class_label_save = class_label.clone()
        cam_save = cam.clone()
        cam_save = compute_cam_up_v2(cam_save, class_label, w // 4, h // 4, b)
        cam_save = cam_save[0]
        # print('cam_save形状', cam_save)
        # print(class_label_save)
        # print(torch.nonzero(class_label_save)[:, 1])
        valid_category = torch.nonzero(class_label_save)[:, 1]

        # print(valid_category)
        cam_save = cam_save[valid_category]

        # print('存储形状', cam_save)
        cam_save /= F.adaptive_max_pool2d(cam_save, (1, 1)) + 1e-5
        # print(cam_save)

        np.save(os.path.join('/media/disk8T/more2/wr/RRM/cam_pred/dict_label', img_name[0].replace('jpg', 'npy')),
                {"keys": valid_category, "cam":cam_save.cpu()})

        # print(cam.shape)
        # print(outputs)
        # print(cam.shape)
        # x_f, cam, seg = model(image_fea, require_seg=True, require_mcam=True)
        cam_up = compute_cam_up(cam, class_label, w, h, b) #获取类激活映射图（b,6,h,w）
        # print('类激活映射', cam_up.shape)
        seg_label = np.zeros((b, w, h))
        cam_weight = np.zeros((b, w, h))
        for i in range(b):
        # print(i)
            cam_up_single = cam_up[i] #取每个patch的第i个图
            cam_label = class_label[i].cpu().numpy()
            # print(type(ori_image))
            ori_image_i = ori_image[i].astype(np.uint8)

            # print('类别尺寸', cam_label)
            # print('原图尺寸', ori_image_i)
            norm_cam = cam_up_single / (np.max(cam_up_single, (1, 2), keepdims=True) + 1e-5) #归一化cam
            # print('激活尺寸', norm_cam)

            seg_label[i] = compute_seg_label_v3(ori_image_i, cam_label, norm_cam) #原图（3，512，512）cam_label标签尺寸（6，），激活尺寸（6， 512， 512）
            seg_label = seg_label[i].reshape(h, w)
            np.save('/media/disk8T/more2/wr/RRM/cam_pred/label/'+ img_name[0] + '.npy', seg_label)
            seg_label_map = visual_v2(seg_label)
            imsave('/media/disk8T/more2/wr/RRM/cam_pred/visual_label/'+ img_name[0] + '.png', seg_label_map)
print('完成')