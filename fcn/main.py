import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
import os
import random
# from dataloader import sarsegDataset
from dataloader4 import sarsegDataset_train, sarsegDataset_test
from torch.utils.data import DataLoader
from model_res50 import FeatureResNet, SegResNet
from torchvision import models
from segmetric import metric, visual
from imageio import imsave
import matplotlib.pyplot as plt

os.environ['TORCH_HOME'] = '/media/disk8T/wr/torch-model'

save_mode_path = '/media/disk8T/more2/wr/RRM/seg/seg_model'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#setup
parser = ArgumentParser(description='SAR Semantic Segmentation')
parser.add_argument('--dataroot', type=str, default='/media/disk8T/wr/PolSAR-Seg41-rot-crop-movezero', help='Dataset root')
parser.add_argument('--pseudo', type=str, default='/media/disk8T/more2/wr/RRM/irn_fuse_pred/label', help='Dataset root')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--workers', type=int, default=1, help='Data loader workers')
parser.add_argument('--epochs', type=int, default=70, help='Training epochs')
parser.add_argument('--lr', type=float, default=0.00003, help='Learning rate')
parser.add_argument('--eps', type=float, default=1e-5, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)

#dataloader
train_dataset = sarsegDataset_train(sar_root=args.dataroot, pseudo_root=args.pseudo, is_train=True)
test_dataset = sarsegDataset_test(sar_root=args.dataroot, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

#training
pretrained_net = FeatureResNet()
pretrained_net.load_state_dict(models.resnet50(pretrained=True).state_dict())
# pretrained_net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
net = SegResNet(args.num_classes, pretrained_net).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# def train(train_loader, model, criterion, optimizer, scheduler, epoches):
#     for epoch in range(epoches+1):
#         total_loss = 0
#         for i, (data, target) in enumerate(train_loader):
#             model.train()
#             scheduler.step()
#             input, label = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(input)
#             loss = criterion(output, label)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()*input.size(0)
#             if i % 10 == 0:
#                 print('[Epoch {}/{}] [iter {}] Loss:{:.4f}'.format(epoch+1, epoches, i+1, total_loss/10))
#                 total_loss = 0

# #训练
train_loss = []
for epoch in range(args.epochs+1):
    total_loss = 0
    for i, (data, target) in enumerate(train_loader):
        net.train()
        # exp_lr_scheduler.step()
        input, label = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*input.size(0)
        if i % 10 == 0:
            print('[Epoch {}/{}] [iter {}] Loss:{:.4f}'.format(epoch+1, args.epochs, i+1, total_loss/10))
            train_loss.append((total_loss/10))
            total_loss = 0

    # if epoch == 50 or epoch == 55:
    #     save_model_path_all = os.path.join(save_mode_path, 'model'+str(epoch)+'.pt')
    #     torch.save(net.state_dict(), save_model_path_all)
    #     print('save model.....')

save_path = os.path.join(save_mode_path, 'allmodel.pt')
torch.save(net.state_dict(), save_path)
#存储train loss
f_trainloss = open('train_loss.txt', 'w')
f_trainloss.write(str(train_loss))
f_trainloss.close()


#测试
# net.load_state_dict(torch.load('/media/disk4T/wr/model/model65.pt'))
net.eval()
    #是否不计算背景像元
segmetric = metric(num_class=args.num_classes)
for i, (image_name, data, target) in enumerate(test_loader):
    input, label = data.to(device), target.to(device)
    output = net(input) #size(1, 7, h, w)
    b,c,h,w = output.size()
    output = output.permute(0, 2, 3, 1).contiguous().view(-1, c)
    logits = F.softmax(output, dim=1).detach().cpu().numpy()
    pred = np.argmax(logits, axis=1).reshape([h, w])
    label = label.cpu().numpy().reshape([h, w])

        #可视化结果
    color_map = visual(pred)
    #print(color_map.shape)
    imsave('/media/disk8T/more2/wr/RRM/seg/map/'+image_name[0]+'.png', color_map)
        #指标计算,将混淆矩阵相加
        #去除背景像元
    pred = pred
    label = label
    # print(label)
    segmetric.addBatch(pred, label)
    #输出指标
#segmetric.outback()
OA = segmetric.PA()
per_class_acc = segmetric.CPA()
AA = segmetric.mPA()
mIoU = segmetric.mIoU()
kappa = segmetric.Kappa()
print('测试结果')
print('OA:{:.4f}'.format(OA))
print('每类精度', per_class_acc)
print('AA:{:.4f}'.format(AA))
print('mIoU:{:.4f}'.format(mIoU))
print('kappa:{:.4f}'.format(kappa))

#记录结果
f = open("result.txt","a")
f.write('\n')
f.write('\n')
f.write('epochs = '+str(args.epochs)+'learning rate = '+ str(args.lr)+'batch size = '+str(args.batch_size))
f.write('OA='+str(OA)+'\n')
f.write('per_class_acc='+str(per_class_acc)+'\n')
f.write('AA='+str(AA)+'\n')
f.write('mIoU='+str(mIoU)+'\n')
f.write('kappa='+str(kappa))
f.close()

# #可视化loss
# def data_read(path):
#     with open(path, 'r') as f:
#         raw_data = f.read()
#         data = raw_data[1:-1].split(',')
    
#     return np.asfarray(data, float)

# train_loss_path = 'train_loss.txt'
# y_train_loss = data_read(train_loss_path)
# x_train_loss = range(len(y_train_loss))
# plt.plot(x_train_loss, y_train_loss, color='b')
# plt.savefig('train_loss.png')



# #测试程序，通过累加混淆矩阵
# def test1(test_loader, model, num_class):
#     model.eval()
#     #是否不计算背景像元
#     segmetric = metric(num_class=num_class)
#     for i, (data, target) in enumerate(test_loader):
#         input, label = data.to(device), target.to(device)
#         output = model(input) #size(1, 7, h, w)
#         b,c,h,w = output.size()
#         output = output.permute(0, 2, 3, 1).contiguous().view(-1, c)
#         logits = F.softmax(output, dim=1).detach().cpu().numpy()
#         pred = np.argmax(logits, axis=1).reshape([h, w])
#         label = label.cpu().numpy().reshape([h, w])

#         #可视化结果
#         if i % 50 == 0:
#             color_map = visual(pred)
#             print(color_map.shape)
#             imsave('color_map'+str(i)+'.png', color_map)
#         #指标计算,将混淆矩阵相加
#         #去除背景像元
#         pred = pred
#         label = label
#         segmetric.addBatch(pred, label)
#     #输出指标
#     segmetric.outback()
#     OA = segmetric.PA()
#     per_class_acc = segmetric.CPA()
#     AA = segmetric.mPA()
#     mIoU = segmetric.mIoU()
#     print('测试结果')
#     print('OA:{:.4f}'.format(OA))
#     print('每类精度', per_class_acc)
#     print('AA:{:.4f}'.format(AA))
#     print('mIoU:{:.4f}'.format(mIoU))


# #测试程序，通过取所有测试样本均值
# def test2(test_loader, model, num_class):
#     model.eval()
#     OA_list = []
#     per_class_acc_list = []
#     AA_list = []
#     mIoU_list = []
#     #是否不计算背景像元
#     for i, (data, target) in enumerate(test_loader):
#         input, label = data, target
#         output = model(input) #size(1, 7, h, w)
#         b,c,h,w = output.size()
#         output = output.permute(0, 2, 3, 1).contiguous().view(-1, c)
#         logits = F.softmax(output, dim=1).numpy()
#         pred = np.argmax(logits, axis=1).reshape([h, w])
#         label = label.numpy().reshape([h, w])

#         #可视化结果
#         if i % 50 == 0:
#             color_map = visual(pred)
#             imsave('color_map'+str(i)+'.png', color_map)

#         #指标计算,将混淆矩阵相加
#         #去除背景像元
#         segmetric = metric(num_class=num_class - 1)
#         pred = pred-1
#         label = label-1
#         segmetric.addBatch(pred, label)

#         OA = segmetric.PA()
#         OA_list.append(OA)
#         per_class_acc = segmetric.CPA()
#         per_class_acc_list.append(per_class_acc)
#         AA = segmetric.mPA()
#         AA_list.append(AA)
#         mIoU = segmetric.mIoU()
#         mIoU_list.append(mIoU)

#     #输出指标
#     print('测试结果')
#     print('OA:{:.4f}'.format(np.mean(OA_list)))
#     print('per_class_acc:{:.4f}'.format(np.mean(per_class_acc_list, axis=0)))
#     print('AA:{:.4f}'.format(np.mean(AA_list)))
#     print('mIoU:{:.4f}'.format(np.mean(mIoU_list)))


# if __name__ == '__main__':
#     train(train_loader, net, criterion, optimizer, exp_lr_scheduler, args.epochs)
#     test1(test_loader, net, args.num_classes)













