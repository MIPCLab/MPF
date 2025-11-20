import torch
from dataloader import sarsegDataset_train, sarsegDataset_test
from misc import pyutils, torchutils, indexing
from argparse import ArgumentParser
import os
import random
from net import resnet50_irn
from dataloader_irn import sarsegDataset_irn, sarsegDataset_infer
from torch.utils.data import DataLoader


os.environ['TORCH_HOME'] = '/media/disk8T/wr/torch-model'


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description='SAR RRM')
parser.add_argument('--dataroot', type=str, default='/media/disk8T/more2/wr/eeds/sar_data', help='Dataset root')
parser.add_argument('--irn_layer1_label_root', type=str, default='/media/disk8T/more2/wr/RRM_eeds/cam/label', help='Dataset layer1 root')
parser.add_argument('--irn_layer2_label_root', type=str, default='/media/disk8T/more2/wr/RRM_eeds/layer_cam/layer3/cam/seg_label', help='Dataset layer2 root')
parser.add_argument('--labelroot', type=str, default='/media/disk8T/more2/wr/eeds/label_data/label_class.npy', help='Dataset root')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--workers', type=int, default=1, help='Data loader workers')
parser.add_argument('--crop_size', type=int, default=256, help='Img size')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--num_cls', type=int, default=6, help='number of classification classes')
parser.add_argument('--num_seg', type=int, default=6, help='number of segmentation classes')
parser.add_argument("--irn_num_epoches", default=5, type=int)
parser.add_argument("--irn_lr", default=0.01, type=float)
parser.add_argument("--irn_wt_dec", default=5e-4, type=float)
parser.add_argument("--session_name", default="SAR-RRM", type=str)
args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)




path_index = indexing.PathIndex(radius=8, default_size=(args.crop_size // 4, args.crop_size // 4))
model = resnet50_irn.AffinityDisplacementLoss(path_index)
train_dataset = sarsegDataset_irn(args.dataroot, args.irn_layer1_label_root, args.irn_layer2_label_root, crop_size=256, indices_from=path_index.src_indices, indices_to=path_index.dst_indices, is_train=True)
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=args.workers)

pyutils.Logger(args.session_name + '.log')

max_step = (train_dataset.__len__() // args.batch_size) * args.irn_num_epoches

param_groups = model.trainable_parameters()
optimizer = torchutils.PolyOptimizer([
    {'params': param_groups[0], 'lr': 1*args.irn_lr, 'weight_decay': args.irn_wt_dec},
    {'params': param_groups[1], 'lr': 10*args.irn_lr, 'weight_decay': args.irn_wt_dec}
], lr=args.irn_lr, weight_decay=args.irn_wt_dec, max_step=max_step)


model = model.to(device)
model.train()

avg_meter = pyutils.AverageMeter()
timer = pyutils.Timer()

for ep in range(args.irn_num_epoches):
    print('Epoch %d/%d' % (ep+1, args.irn_num_epoches))

    for iter, pack in enumerate(train_data_loader):
        image_name, sar_feature, irn_label_layer1, irn_label_layer2, aff_fg_pos_label, aff_neg_label = pack
        sar_feature = sar_feature.to(device)
        irn_label_layer1 = irn_label_layer1.to(device)
        irn_label_layer2 = irn_label_layer2.to(device)
        fg_pos_label = aff_fg_pos_label.to(device)
        neg_label = aff_neg_label.to(device)

        pos_aff_loss, neg_aff_loss, dp_fg_loss = model(sar_feature, True)

        fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)
        pos_aff_loss = fg_pos_aff_loss
        neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

        dp_fg_loss = torch.sum(dp_fg_loss * torch.unsqueeze(fg_pos_label, 1)) / (2 * torch.sum(fg_pos_label) + 1e-5)

        avg_meter.add({'loss1': pos_aff_loss.item(), 'loss2': neg_aff_loss.item(),
                           'loss3': dp_fg_loss.item()})
        
        total_loss = pos_aff_loss + neg_aff_loss + dp_fg_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (optimizer.global_step - 1) % 50 == 0:
            timer.update_progress(optimizer.global_step / max_step)

            print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                  'loss:%.4f %.4f %.4f' % (
                      avg_meter.pop('loss1'), avg_meter.pop('loss2'), avg_meter.pop('loss3')),
                      'imps:%.1f' % ((iter + 1) * args.batch_size / timer.get_stage_elapsed()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']),
                      'etc:%s' % (timer.str_estimated_complete()), flush=True)
        else:
            timer.reset_stage()


#推断位移
save_path = '/media/disk8T/more2/wr/RRM_eeds/RRM/model'
infer_dataset = sarsegDataset_infer(args.dataroot, is_train=True)
infer_data_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

model.eval()
print('分析移位均值', end='')
dp_mean_list = []

with torch.no_grad():
    for i, pack in enumerate(infer_data_loader):
        image_name, sar_feature = pack
        sar_feature = sar_feature.to(device)
        aff, dp = model(sar_feature, False)
        dp_mean_list.append(torch.mean(dp, dim=(0, 2, 3)).cpu())
    model.mean_shift.running_mean = torch.mean(torch.stack(dp_mean_list), dim=0)
print('done.')

torch.save(model.state_dict(), os.path.join(save_path, 'irn.pth'))
torch.cuda.empty_cache()

