import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


#2D batch normalization
def bn(planes):
    layer = nn.BatchNorm2d(planes)
    init.constant(layer.weight, 1)
    init.constant(layer.bias, 0)

    return layer

#2D space-preserving padding
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
    if transposed:
        layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1, dilation=dilation, bias=bias)
        #双线性插值法初始化反卷积核
        w = torch.Tensor(kernel_size, kernel_size)
        centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
        for y in range(kernel_size):
            for x in range(kernel_size):
                w[y, x] = (1 - abs((x-centre)/stride)) * (1 - abs((y-centre)/stride))
        layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
    else:
        padding = (kernel_size + 2*(dilation - 1)) // 2
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    if bias:
        init.constant(layer.bias, 0)

    return layer


class FeatureResNet(ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 6, 3], 1000)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.bn1(x1)
        x = self.relu(x)
        x2 = self.maxpool(x)
        x = self.layer1(x2)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        #x5的尺寸为512*16*16

        return x1, x2, x3, x4, x5


class Net(FeatureResNet):
    def __init__(self, num_classes, pretrain_net):
        super().__init__()

        self.pretrain_net = pretrain_net
        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(2048, num_classes, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)

        self.not_training = []
        self.from_scratch_layers = [self.fc8]



    def forward(self, x):
        x1, x2, x3, x4, x5 = self.pretrain_net(x)

        x5 = self.dropout7(x5)
        x5 = F.avg_pool2d(x5, kernel_size=(x5.size(2), x5.size(3)), padding=0)
        x5 = self.fc8(x5)
        x5 = x5.view(x5.size(0), -1)

        return x5

    def get_parameter_groups(self):

        groups = ([], [], [], [])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups





# class Net(FeatureResNet):
#     def __init__(self, num_classes, pretrained_net):
#         super().__init__()
#         self.pretrained_net = pretrained_net
#
#         self.dropout7 = torch.nn.Dropout2d(0.5)
#
#         self.fc8 = nn.Conv2d(512, num_classes, 1, bias=False)
#         torch.nn.init.xavier_uniform_(self.fc8.weight)
#
#         self.fc8_seg_conv1 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=12, dilation=12, bias=True)
#         torch.nn.init.xavier_uniform_(self.fc8_seg_conv1.weight)
#
#         self.fc8_seg_conv2 = nn.Conv2d(512, num_classes, (3, 3), stride=1, padding=12, dilation=12, bias=True)
#         torch.nn.init.xavier_uniform_(self.fc8_seg_conv2.weight)
#
#
#         self.not_training = []
#         self.from_scratch_layers = [self.fc8, self.fc8_seg_conv1, self.fc8_seg_conv2]
#
#         # self.not_training = []
#         # self.from_scratch_layers = [self.fc8, self.fc8_seg_conv1, self.fc8_seg_conv2, self.fc8_seg_conv1_tp, self.fc8_seg_conv2_tp, self.fc8_seg_conv3_tp, self.fc8_seg_conv4_tp, self.fc8_seg_conv5_tp, self.conv10]
#
#
#     def forward(self, x, require_seg=True, require_mcam=True):
#         x = self.pretrained_net(x)
#         if require_seg == True and require_mcam == True:
#             x_cam = x.clone()
#             x_seg = x.clone()
#
#             x = self.dropout7(x)
#             x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0) #全局平均池化
#
#             x = self.fc8(x)
#             x = x.view(x.size(0), -1) #(b,6)类别输出
#
#             cam = F.conv2d(x_cam, self.fc8.weight)
#             cam = F.relu(cam)
#
#             #x_seg的形状（4,512,8,8）
#             # print('输出x_seg尺寸', x_seg.shape)
#             x_seg = self.fc8_seg_conv1(x_seg)
#             x_seg = F.relu(x_seg)
#             x_seg = self.fc8_seg_conv2(x_seg)
#
#             # ##new seg_head
#             # x_seg = self.relu(self.seg_bn1(self.fc8_seg_conv1_tp(x_seg)))
#             # x_seg = self.relu(self.seg_bn2(self.fc8_seg_conv2_tp(x_seg + x4)))
#             # x_seg = self.relu(self.seg_bn3(self.fc8_seg_conv3_tp(x_seg + x3)))
#             # x_seg = self.relu(self.seg_bn4(self.fc8_seg_conv4_tp(x_seg + x2)))
#             # x_seg = self.relu(self.seg_bn5(self.fc8_seg_conv5_tp(x_seg + x1)))
#             # x_seg = self.conv10(x_seg)
#
#             return x, cam, x_seg
#
#         elif require_mcam == True and require_seg == False:
#             x_cam = x.clone()
#             cam = F.conv2d(x_cam, self.fc8.weight)
#             cam = F.relu(cam)
#
#             return cam
#
#         else:
#             x = self.dropout7(x)
#             x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
#             x = self.fc8(x)
#             x = x.view(x.size(0), -1)
#
#             return x
#
#     def forward_cam(self, x):
#         x = self.pretrained_net(x)
#
#         x = F.conv2d(x, self.fc8.weight)
#         x = F.relu(x)
#
#         return x
#
#     def get_parameter_groups(self):
#         groups = ([], [], [], [])
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.weight.requires_grad:
#                     if m in self.from_scratch_layers:
#                         groups[2].append(m.weight)
#                     else:
#                         groups[0].append(m.weight)
#
#                 if m.bias is not None and m.bias.requires_grad:
#
#                     if m in self.from_scratch_layers:
#                         groups[3].append(m.bias)
#                     else:
#                         groups[1].append(m.bias)
#
#         return groups

class Net_CAM(Net):
    def __init__(self, num_classes, pretrain_net):
        super(Net_CAM, self).__init__(num_classes, pretrain_net)

    def forward(self, x):
        x = self.pretrain_net(x)
        x_feature = x.clone()
        x = self.dropout7(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = self.fc8(x)
        x = x.view(x.size(0), -1)

        cams = F.conv2d(x_feature, self.fc8.weight)
        cams = F.relu(cams)

        return x, cams, x_feature


class Net_CAM_Feature(Net):
    def __init__(self, num_classes, pretrain_net):
        super(Net_CAM_Feature, self).__init__(num_classes, pretrain_net)


    def forward(self, x):

        input_size_h = x.size()[2]
        input_size_x = x.size()[3]

        x = self.pretrain_net(x)
        x_feature = x.clone()
        x = self.dropout7(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = self.fc8(x)
        x = x.view(x.size(0), -1)

        cams = F.conv2d(x_feature, self.fc8.weight) #输出尺寸为（b,20,w,h）
        cams = F.relu(cams) #输出尺寸为（b,20,w,h）
        cams = cams/(F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2)*x_feature.unsqueeze(1)
        cams_feature = cams_feature.view(cams_feature.size(0), cams_feature.size(1), cams_feature.size(2), -1)#(b,20,2048,32*32)
        cams_feature = torch.mean(cams_feature, -1) #生成cam_feature (b,20,2048,1)

        return x, cams_feature, cams
    
    # def forward(self, x):

    #     input_size_h = x.size()[2]
    #     input_size_x = x.size()[3]
    #     x_ori = x.clone()

    #     x = self.pretrain_net(x)
    #     x_feature = x.clone()
    #     x = self.dropout7(x)
    #     x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
    #     x = self.fc8(x)
    #     x = x.view(x.size(0), -1)

    #     cams = F.conv2d(x_feature, self.fc8.weight) #输出尺寸为（b,20,w,h）
    #     cams = F.relu(cams) #输出尺寸为（b,20,w,h）
    #     cams = cams/(F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
    #     _,_,w,h = x_feature.shape
    #     x_ori_feature = F.interpolate(x_ori, size=(w,h), mode='bilinear')
    #     cams_feature = cams.unsqueeze(2)*x_ori_feature.unsqueeze(1)
    #     cams_feature = cams_feature.view(cams_feature.size(0), cams_feature.size(1), cams_feature.size(2), -1)#(b,20,4,32*32)
    #     cams_feature = torch.mean(cams_feature, -1) #生成cam_feature (b,20,4,1)

    #     return x, cams_feature, cams


class CAM(Net_CAM_Feature):
    def __init__(self, num_classes, pretrain_net):
        super(CAM, self).__init__(num_classes, pretrain_net)

    def forward(self, x, separate=False):
        x1, x2, x3, x4, x5 = self.pretrain_net(x)
        x5 = F.conv2d(x5, self.fc8.weight)
        if separate:
            return x5
        x5 = F.relu(x5)
        # x = x[0] + x[1].flip(-1)

        return x5

    def forward1(self, x, weight, separate=False):
        x = self.pretrain_net(x)
        x = F.conv2d(x, weight)

        if separate:
            return x

        x = F.relu(x)
        # x = x[0] + x[1].flip(-1)

        return x

    def forward2(self, x, weight, separate=False):
        x = self.pretrain_net(x)
        x = F.conv2d(x, weight*self.fc8.weight)

        if separate:
            return x
        x = F.relu(x)
        # print(x)
        # x = x[0] + x[1].flip(-1)

        return x


class Class_Predictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(representation_size, num_classes, 1, bias=False)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_classes, -1) #维度为（b,20,2048）
        mask = label>0

        feature_list = [x[i][mask[i]] for i in range(batch_size)] #尺寸为[(n1,2048),...,(nbs,2048)]
        prediction = [self.classifier(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]


        loss = 0
        acc = 0
        num = 0
        for logit, label in zip(prediction, labels):
            if label.shape[0] == 0:
                continue

            loss_ce = F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1) == label.view(-1)).sum().float()
            num += label.size(0)

        return loss/batch_size, acc/num


# class SegNet(Net):
#     def __init__(self, num_classes, pretrained_net):
#         super().__init__(num_classes, pretrained_net)
#
#     def forward(self, x, require_seg=True, require_mcam=True):
#         if require_seg == True and require_mcam == True:
#             input_size_h = x.size()[2]
#             input_size_w = x.size()[3]
#
#             x2 = F.interpolate(x, size=(int(input_size_h * 0.5), int(input_size_w * 0.5)), mode='bilinear', align_corners=False)
#             x3 = F.interpolate(x, size=(int(input_size_h * 1.5), int(input_size_w * 1.5)), mode='bilinear', align_corners=False)
#             x4 = F.interpolate(x, size=(int(input_size_h * 2), int(input_size_w * 2)), mode='bilinear', align_corners=False)
#
#             seg = []
#             #控制作用域
#             with torch.enable_grad():
#                 xf1, cam1, seg1 = super().forward(x, require_seg=True, require_mcam=True)
#             with torch.no_grad():
#                 cam2 = super().forward(x2, require_seg=False, require_mcam=True)
#                 cam3 = super().forward(x3, require_seg=False, require_mcam=True)
#                 cam4 = super().forward(x4, require_seg=False, require_mcam=True)
#
#             xf_temp = xf1
#
#             cam2 = F.interpolate(cam2, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear', align_corners=False)
#             cam3 = F.interpolate(cam3, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear', align_corners=False)
#             cam4 = F.interpolate(cam4, size=(int(seg1.shape[2]), int(seg1.shape[3])), mode='bilinear', align_corners=False)
#             # cam2 = F.interpolate(cam2, size=(int(cam1.shape[2]), int(cam1.shape[3])), mode='bilinear', align_corners=False)
#             # cam3 = F.interpolate(cam3, size=(int(cam1.shape[2]), int(cam1.shape[3])), mode='bilinear', align_corners=False)
#             # cam4 = F.interpolate(cam4, size=(int(cam1.shape[2]), int(cam1.shape[3])), mode='bilinear', align_corners=False)
#
#             cam = (cam1+cam2+cam3+cam4)/4
#             # cam = cam1
#             seg.append(seg1)
#
#             return xf_temp, cam, seg
#
#         if require_mcam == False and require_seg == False:
#             xf = super().forward(x, require_seg=False, require_mcam=False)
#             self.not_training = [self.conv1]
#
#             return xf
#
#         # if require_mcam == False and require_seg == True:
#         #     xf, cam, seg = super().forward(x, require_seg=True, require_mcam=True)
#
#         if require_mcam == False and require_seg == True:
#             xf, cam, seg = super().forward(x, require_seg=True, require_mcam=True)
#
#             return seg
#
#
#     def get_parameter_groups(self):
#         groups = ([], [], [], [])
#
#         for m in self.modules():
#
#             if isinstance(m, nn.Conv2d):
#
#                 if m.weight.requires_grad:
#                     if m in self.from_scratch_layers:
#                         groups[2].append(m.weight)
#                     else:
#                         groups[0].append(m.weight)
#
#                 if m.bias is not None and m.bias.requires_grad:
#
#                     if m in self.from_scratch_layers:
#                         groups[3].append(m.bias)
#                     else:
#                         groups[1].append(m.bias)
#
#         return groups