import numpy as np
import tool.imutils as imutils
import torch
import torch.nn.functional as F
import cv2
import random
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def _crf_with_alpha(ori_img, cam_dict, alpha):
    v = np.array(list(cam_dict.values()))
    h, w, c = ori_img.shape
    # print(v.shape)
    # print(np.max(v, axis=0, keepdims=True))
    if v.shape[0] == 0:
        bg_score = np.ones([h, w])
        bgcam_score = np.expand_dims(bg_score, axis=0)

        n_crf_al = np.zeros([7, bgcam_score.shape[1], bgcam_score.shape[2]])
        n_crf_al[0, :, :] = bgcam_score[0, :, :] 
    else:
        bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha) #背景，对于激活值大的变小，作为背景，类激活值大即可代表作为背景的可能性越小，alpha用于调节bg_score打分
        bgcam_score = np.concatenate((bg_score, v), axis=0) #获取背景cam评分(7,h,w)
        # print('类别数', bgcam_score.shape)
        crf_score = imutils.crf_inference(ori_img, bgcam_score, labels=bgcam_score.shape[0]) #经过crf处理后的cam结果

        n_crf_al = np.zeros([7, bg_score.shape[1], bg_score.shape[2]])
        n_crf_al[0, :, :] = crf_score[0, :, :] #将crf处理得分后的结果中的背景类赋予给n_crf
        for i, key in enumerate(cam_dict.keys()):
            n_crf_al[key + 1] = crf_score[i + 1]
        
    return n_crf_al

    # print(v.shape)
    # bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha) #背景，对于激活值大的变小，作为背景，类激活值大即可代表作为背景的可能性越小，alpha用于调节bg_score打分
    # bgcam_score = np.concatenate((bg_score, v), axis=0) #获取背景cam评分
    # # print('类别数', bgcam_score.shape)
    # crf_score = imutils.crf_inference(ori_img, bgcam_score, labels=bgcam_score.shape[0]) #经过crf处理后的cam结果

    # n_crf_al = np.zeros([7, bg_score.shape[1], bg_score.shape[2]])
    # n_crf_al[0, :, :] = crf_score[0, :, :] #将crf处理得分后的结果中的背景类赋予给n_crf
    # for i, key in enumerate(cam_dict.keys()):
    #     n_crf_al[key + 1] = crf_score[i + 1]

    # return n_crf_al


def compute_seg_label(ori_img, cam_label, norm_cam):
    cam_label = cam_label.astype(np.uint8) #one-hot标签

    cam_dict = {}
    cam_np = np.zeros_like(norm_cam) #norm_cam尺寸为(c, h, w)
    for i in range(6):
        if cam_label[i] > 1e-5: #循环每个标签如果标签不为0
            cam_dict[i] = norm_cam[i] #将有标签的每个cam_map所对应的类别通道的norm_cam提取出来,将每个类别的norm_cam提取出来,存入字典中
            cam_np[i] = norm_cam[i] #将cam值存入cam_up中

    bg_score = np.power(1 - np.max(cam_np, 0), 90) #np.max(cam_up, 0)即选择每个cam所在像素点通道最大的值,求次幂；若cam值比较大，则背景的概率就越小，每个像素判断为背景的概率就越小
    bg_score = np.expand_dims(bg_score, axis=0) #转为（1, h, w)，扩展通道维度，如果cam比较大，背景元概率越小
    cam_all = np.concatenate((bg_score, cam_np)) #添加背景cam
    _, bg_w, bg_h = bg_score.shape

    cam_img = np.argmax(cam_all, 0) #取通道最大，生成cam_img
    # print('预测标签', np.unique(cam_img))
    # print('cam键值', cam_dict.keys())
    # print('原图尺寸', ori_img.shape)

    crf_la = _crf_with_alpha(ori_img, cam_dict, 50) #ori_img (w, h, c) cam值的字典 输出cam经过crf处理后的结果, 弱背景
    crf_ha = _crf_with_alpha(ori_img, cam_dict, 110) #ori_img (w, h, c) cam值的字典 输出cam经过crf处理后的结果，强背景，削弱背景影响
    crf_la_label = np.argmax(crf_la, 0) #取la_通道最大
    crf_ha_label = np.argmax(crf_ha, 0) #取ha_通道最小
    # print('crf_la预测标签', np.unique(crf_la_label))
    # print('crf_ha预测标签', np.unique(crf_ha_label))
    crf_label = crf_la_label.copy()
    crf_label[crf_la_label == 0] = 255 #将背景像元值改为255
    crf_label_ori = crf_label.copy()
    # print('crf预测标签', np.unique(crf_label_ori))

    single_img_classes = np.unique(crf_la_label) #生成CAM的标，签不是one-hot类型
    cam_sure_region = np.zeros([bg_w, bg_h], dtype=bool) #cam置信，选取置信度高的区域
    for class_i in single_img_classes:
        if class_i != 0:
            class_not_region = (cam_img != class_i) #对于预测的第i类，在选出原cam_img中不属于这个类的区域
            cam_class = cam_all[class_i, :, :]  #取出该类别所对应的类激活值，第i类对应的类激活值
            cam_class[class_not_region] = 0 #将不属于这个类别的类激活值变为0，不属于这个类别的设为0
            cam_class_order = cam_class[cam_class > 0.1] #选出cam值大于0.1的区域，输出cam中大于0.1的值
            cam_class_order = np.sort(cam_class_order) #对cam值的大小进行排序
            confidence_pos = int(cam_class_order.shape[0] * 0.5) #取0.6分位
            confidence_value = cam_class_order[confidence_pos] #取0.6分位的cam值，原本所对应的类激活map中，将预测为第i类的区域选出，然后找出类别对应的cam_map，再次选择置信区域
            class_sure_region = (cam_class > confidence_value) #找出确信区域 将置信的区域归置为True
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region) #逻辑或,选出置信区域
        else:
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            class_sure_region = (cam_class > 0.2)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)

    cam_not_sure_region = ~cam_sure_region #取非号，选出不确定区域

    crf_label[crf_ha_label == 0] = 0 #设置背景像元
    crf_label_np = np.concatenate([np.expand_dims(crf_ha[0, :, :], axis=0), crf_la[1:, :, :]]) #将ha背景与la目标cam拼接
    crf_not_sure_region = np.max(crf_label_np, 0) < 0.3  #将小于0.8的设置为crf不确定区域
    not_sure_region = np.logical_or(crf_not_sure_region, cam_not_sure_region) #crf和cam均存在不确定区域即设为不确定区域

    crf_label[not_sure_region] = 255 #将不确定区域设为255
    # print('最终预测标签', np.unique(crf_label))

    return crf_label


def compute_seg_label_v2(ori_img, cam_label, norm_cam):
    cam_label = cam_label.astype(np.uint8) #one-hot标签

    cam_dict = {}
    cam_np = np.zeros_like(norm_cam) #norm_cam尺寸为(c, h, w)
    for i in range(6):
        if cam_label[i] > 1e-5: #循环每个标签如果标签不为0
            cam_dict[i] = norm_cam[i] #将有标签的每个cam_map所对应的类别通道的norm_cam提取出来,将每个类别的norm_cam提取出来,存入字典中
            cam_np[i] = norm_cam[i] #将cam值存入cam_up中

    # bg_score = np.power(1 - np.max(cam_np, 0), 100) #np.max(cam_up, 0)即选择每个cam所在像素点通道最大的值,求次幂；若cam值比较大，则背景的概率就越小，每个像素判断为背景的概率就越小
    # bg_score = np.expand_dims(bg_score, axis=0) #转为（1, h, w)，扩展通道维度，如果cam比较大，背景元概率越小
    # cam_all = np.concatenate((bg_score, cam_np)) #添加背景cam
    _, bg_w, bg_h = cam_np.shape
    cam_all = cam_np

    cam_img = np.argmax(cam_all, 0) #取通道最大，生成cam_img
    # print('预测标签', np.unique(cam_img))
    # print('cam键值', cam_dict.keys())
    cam_label = cam_img.copy()
    # print('原图尺寸', ori_img.shape)

    #不使用crf
    # crf_la = _crf_with_alpha(ori_img, cam_dict, 50) #ori_img (w, h, c) cam值的字典 输出cam经过crf处理后的结果, 弱背景
    # crf_ha = _crf_with_alpha(ori_img, cam_dict, 110) #ori_img (w, h, c) cam值的字典 输出cam经过crf处理后的结果，强背景，削弱背景影响
    # crf_la_label = np.argmax(crf_la, 0) #取la_通道最大
    # crf_ha_label = np.argmax(crf_ha, 0) #取ha_通道最小
    # print('crf_la预测标签', np.unique(crf_la_label))
    # print('crf_ha预测标签', np.unique(crf_ha_label))
    # crf_label = crf_la_label.copy()
    # crf_label[crf_la_label == 0] = 255 #将背景像元值改为255
    # crf_label_ori = crf_label.copy()
    # print('crf预测标签', np.unique(crf_label_ori))
    single_img_classes = np.unique(cam_img)
    cam_sure_region = np.zeros([bg_w, bg_h], dtype=bool)
    for class_i in single_img_classes:
        if class_i != 0:
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            cutoff_ori = 0.1
            if cam_class.max() < cutoff_ori:
                cutoff = cam_class.min()
            else:
                cutoff = cutoff_ori
            cam_class_order = cam_class[cam_class > cutoff]
            cam_class_order = np.sort(cam_class_order)
            # print(cam_class_order.shape)
            confidence_pos = int(cam_class_order.shape[0] * 0.7)
            # print(confidence_pos)
            confidence_value = cam_class_order[confidence_pos]
            class_sure_region = (cam_class > confidence_value)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
        else:
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]
            cam_class[class_not_region] = 0
            class_sure_region = (cam_class > 0.3)
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
    
    cam_not_sure_region = ~cam_sure_region
    not_sure_region = cam_not_sure_region
    cam_label[not_sure_region] = 255
    # print('最终预测', np.unique(cam_label))

    return cam_label


def compute_seg_label_v3(ori_img, cam_label, norm_cam):
    cam_label = cam_label.astype(np.uint8) #one-hot标签

    cam_dict = {}
    cam_np = np.zeros_like(norm_cam) #norm_cam尺寸为(c, h, w)
    for i in range(6):
        if cam_label[i] > 1e-5: #循环每个标签如果标签不为0
            cam_dict[i] = norm_cam[i] #将有标签的每个cam_map所对应的类别通道的norm_cam提取出来,将每个类别的norm_cam提取出来,存入字典中
            cam_np[i] = norm_cam[i] #将cam值存入cam_up中

    # bg_score = np.power(1 - np.max(cam_np, 0), 100) #np.max(cam_up, 0)即选择每个cam所在像素点通道最大的值,求次幂；若cam值比较大，则背景的概率就越小，每个像素判断为背景的概率就越小
    # bg_score = np.expand_dims(bg_score, axis=0) #转为（1, h, w)，扩展通道维度，如果cam比较大，背景元概率越小
    # cam_all = np.concatenate((bg_score, cam_np)) #添加背景cam
    _, bg_w, bg_h = cam_np.shape
    cam_all = cam_np

    cam_img = np.argmax(cam_all, 0) #取通道最大，生成cam_img
    # print('预测标签', np.unique(cam_img))
    # print('cam键值', cam_dict.keys())
    cam_label = cam_img.copy()
    # print('原图尺寸', ori_img.shape)

    #不使用crf
    # crf_la = _crf_with_alpha(ori_img, cam_dict, 50) #ori_img (w, h, c) cam值的字典 输出cam经过crf处理后的结果, 弱背景
    # crf_ha = _crf_with_alpha(ori_img, cam_dict, 110) #ori_img (w, h, c) cam值的字典 输出cam经过crf处理后的结果，强背景，削弱背景影响
    # crf_la_label = np.argmax(crf_la, 0) #取la_通道最大
    # crf_ha_label = np.argmax(crf_ha, 0) #取ha_通道最小
    # print('crf_la预测标签', np.unique(crf_la_label))
    # print('crf_ha预测标签', np.unique(crf_ha_label))
    # crf_label = crf_la_label.copy()
    # crf_label[crf_la_label == 0] = 255 #将背景像元值改为255
    # crf_label_ori = crf_label.copy()
    # print('crf预测标签', np.unique(crf_label_ori))
    single_img_classes = np.unique(cam_img)
    cam_sure_region = np.zeros([bg_w, bg_h], dtype=bool)
    for class_i in single_img_classes:
        class_not_region = (cam_img != class_i)
        cam_class = cam_all[class_i, :, :]
        cam_class[class_not_region] = 0
        cutoff_ori = 0.02
        if cam_class.max() < cutoff_ori:
            cutoff = cam_class.min()
        else:
            cutoff = cutoff_ori
        # print('最小值', cam_class.min())
        cam_class_order = cam_class[cam_class >= cutoff]
        cam_class_order = np.sort(cam_class_order)
        # print(cam_class_order.shape)
        confidence_pos = int(cam_class_order.shape[0] * 0.3)  #0.2的时候最佳
        # print(confidence_pos)
        confidence_value = cam_class_order[confidence_pos]
        class_sure_region = (cam_class > confidence_value)
        cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
        # else:
        #     class_not_region = (cam_img != class_i)
        #     cam_class = cam_all[class_i, :, :]
        #     cam_class[class_not_region] = 0
        #     class_sure_region = (cam_class > 0.3)
        #     cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
    
    cam_not_sure_region = ~cam_sure_region
    not_sure_region = cam_not_sure_region
    cam_label[not_sure_region] = 255
    # print('最终预测', np.unique(cam_label))

    return cam_label


def compute_joint_loss(ori_img, seg, seg_label, croppings, critersion, DenseEnergyLosslayer):
    seg_label = np.expand_dims(seg_label,axis=1) #扩展通道维，（b, 1, h, w)
    seg_label = torch.from_numpy(seg_label)

    w = seg_label.shape[2]
    h = seg_label.shape[3]
    pred = F.interpolate(seg, (w,h), mode="bilinear", align_corners=False)
    pred_softmax = torch.nn.Softmax(dim=1)
    pred_probs = pred_softmax(pred) #语义分割部分softmax预测结果
    ori_img = torch.from_numpy(ori_img.astype(np.float32))
    croppings = torch.from_numpy(croppings.astype(np.float32).transpose(2,0,1)) #格式为(b,h,w)全为1矩阵
    dloss = DenseEnergyLosslayer(ori_img, pred_probs, croppings, seg_label)
    dloss = dloss.cuda()

    seg_label_tensor = seg_label.long().cuda() #生成的伪标签

    seg_label_copy = torch.squeeze(seg_label_tensor.clone()) #伪标签(b, h, w)
    bg_label = seg_label_copy.clone() #背景标签
    fg_label = seg_label_copy.clone() #前景标签
    bg_label[seg_label_copy != 0] = 255 #背景标签，伪标签不为0处,都设为255
    fg_label[seg_label_copy == 0] = 255 #前景标签，伪标签为0处， 设为255
    bg_celoss = critersion(pred, bg_label.long().to(device)) #交叉熵损失

    fg_celoss = critersion(pred, fg_label.long().to(device))

    celoss = bg_celoss + fg_celoss

    return celoss, dloss

def compute_joint_loss_SAR(ori_img, seg, seg_label, critersion):
    seg_label = np.expand_dims(seg_label,axis=1) #扩展通道维，（b, 1, h, w)
    seg_label = torch.from_numpy(seg_label)

    w = seg_label.shape[2]
    h = seg_label.shape[3]
    pred = F.interpolate(seg, (w,h), mode="bilinear", align_corners=False)
    # pred = seg
    pred_softmax = torch.nn.Softmax(dim=1)
    # pred_probs = pred_softmax(pred) #语义分割部分softmax预测结果
    # ori_img = ori_img
    # #ori_img = torch.from_numpy(ori_img.astype(np.float32))
    #croppings = torch.from_numpy(croppings.astype(np.float32).transpose(2,0,1)) #格式为(b,h,w)全为1矩阵
    # dloss = DenseEnergyLosslayer(ori_img, pred_probs, croppings, seg_label)
    # dloss = dloss.cuda()

    seg_label_tensor = seg_label.long().to(device) #生成的伪标签

    seg_label_copy = torch.squeeze(seg_label_tensor.clone()) #伪标签(b, h, w)
    bg_label = seg_label_copy.clone() #背景标签
    fg_label = seg_label_copy.clone() #前景标签
    bg_label[seg_label_copy != 0] = 255 #背景标签，伪标签不为0处,都设为255
    # fg_label[seg_label_copy == 0] = 255 #前景标签，伪标签为0处， 设为255
    # bg_celoss = critersion(pred, bg_label.long().to(device)) #交叉熵损失

    fg_celoss = critersion(pred, fg_label.long().to(device))

    # celoss = bg_celoss + fg_celoss
    celoss = fg_celoss
    # print('bg_celoss', bg_celoss)
    # print('fg_celoss', fg_celoss)

    return celoss


def compute_cam_up_v2(cam, label, w, h, b):
    cam_up = F.interpolate(cam, (w, h), mode='bilinear', align_corners=False)
    # print(label.shape)
    # cam_up = cam_up * label.clone().view(b, 6, 1, 1) #将类别没标签处的map，将类别标签为0处，map变为0
    cam_up = cam_up
    return cam_up

def compute_cam_up(cam, label, w, h, b):
    cam_up = F.interpolate(cam, (w, h), mode='bilinear', align_corners=False)
    # print(label.shape)
    cam_up = cam_up * label.clone().view(b, 6, 1, 1) #将类别没标签处的map，将类别标签为0处，map变为0
    cam_up = cam_up.cpu().data.numpy()
    return cam_up


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def resize_label_batch(label, size):
    label_resized = np.zeros((size, size, 1, label.shape[3]))
    interp = torch.nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = torch.autograd.Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)
    label_resized[label_resized>21] = 255
    return label_resized


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def scale_gt(img_temp, scale):
    new_dims = (int(img_temp.shape[1] * scale), int(img_temp.shape[0] * scale))
    return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST).astype(float)

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]


def RandomCrop(imgarr, cropsize):

    h, w, c = imgarr.shape

    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space+1)
    else:
        cont_left = random.randrange(-w_space+1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space+1)
    else:
        cont_top = random.randrange(-h_space+1)
        img_top = 0

    img_container = np.zeros((cropsize, cropsize, imgarr.shape[-1]), np.float32)

    cropping =  np.zeros((cropsize, cropsize), np.bool)


    img_container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
        imgarr[img_top:img_top+ch, img_left:img_left+cw]
    cropping[cont_top:cont_top + ch, cont_left:cont_left + cw] = 1

    return img_container, cropping

def get_data_from_chunk_v2(chunk, args):
    img_path = args.IMpath

    scale = np.random.uniform(0.7, 1.3)
    dim = args.crop_size
    images = np.zeros((dim, dim, 3, len(chunk)))
    ori_images = np.zeros((dim, dim, 3, len(chunk)),dtype=np.uint8)
    croppings = np.zeros((dim, dim, len(chunk)))
    labels = load_image_label_list_from_npy(chunk)
    labels = torch.from_numpy(np.array(labels))

    for i, piece in enumerate(chunk):
        flip_p = np.random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(img_path, piece + '.jpg'))
        img_temp = cv2.cvtColor(img_temp,cv2.COLOR_BGR2RGB).astype(np.float)
        img_temp = scale_im(img_temp, scale)
        img_temp = flip(img_temp, flip_p)
        img_temp[:, :, 0] = (img_temp[:, :, 0] / 255. - 0.485) / 0.229
        img_temp[:, :, 1] = (img_temp[:, :, 1] / 255. - 0.456) / 0.224
        img_temp[:, :, 2] = (img_temp[:, :, 2] / 255. - 0.406) / 0.225
        img_temp, cropping = RandomCrop(img_temp, dim)
        ori_temp = np.zeros_like(img_temp)
        ori_temp[:, :, 0] = (img_temp[:, :, 0] * 0.229 + 0.485) * 255.
        ori_temp[:, :, 1] = (img_temp[:, :, 1] * 0.224 + 0.456) * 255.
        ori_temp[:, :, 2] = (img_temp[:, :, 2] * 0.225 + 0.406) * 255.
        ori_images[:, :, :, i] = ori_temp.astype(np.uint8)
        croppings[:,:,i] = cropping.astype(np.float32)

        images[:, :, :, i] = img_temp

    images = images.transpose((3, 2, 0, 1))
    ori_images = ori_images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    return images, ori_images, labels, croppings


def compute_cos(fts1, fts2):
    fts1_norm2 = torch.norm(fts1, 2, 1).view(-1, 1)
    fts2_norm2 = torch.norm(fts2, 2, 1).view(-1, 1)

    fts_cos = torch.div(torch.mm(fts1, fts2.t()), torch.mm(fts1_norm2, fts2_norm2.t()) + 1e-7)

    return fts_cos


def compute_dis_no_batch(seg, seg_feature):
    seg = torch.argmax(seg, dim=1, keepdim=True).view(seg.shape[0],1, -1)
    seg_no_batch = seg.permute(0,2,1).clone().view(-1,1)

    bg_label = torch.zeros_like(seg).float()

    bg_label[seg == 0] = 1
    bg_num = torch.sum(bg_label) + 1e-7

    seg_feature = seg_feature.view(seg_feature.shape[0], seg_feature.shape[1], -1)

    seg_feature_no_batch = seg_feature.permute(0, 2, 1).clone()
    seg_feature_no_batch = seg_feature_no_batch.view(-1, seg_feature.shape[1])

    seg_feature_bg = seg_feature * bg_label
    bg_num_batch = torch.sum(bg_label, dim=2)+1e-7
    seg_feature_bg_center = torch.sum(seg_feature_bg, dim=2) / bg_num_batch
    pixel_dis = 0

    bg_center_num = 0
    for batch_i in range(seg_feature.shape[0]):
        bg_num_batch_i = bg_num_batch[batch_i]
        bg_pixel_dis = 1-compute_cos(seg_feature[batch_i].transpose(1,0), seg_feature_bg_center[batch_i].unsqueeze(dim=0))
        if bg_num_batch_i>=1:
            pixel_dis += (torch.sum(bg_pixel_dis * bg_label[batch_i].transpose(1,0), dim=0)/ bg_num_batch_i)
        else:
            pixel_dis += 2*torch.ones([1]).cuda()

        bg_center_num+=1

    fg_center_num=0
    seg_feature_fg_center = torch.zeros([1, 1024])
    batch_num = 0
    for i in range(1, 21):
        class_label = torch.zeros_like(seg_no_batch).float()
        class_label[seg_no_batch == i] = 1
        class_num = torch.sum(class_label) + 1e-7
        batch_num += class_num
        if class_num < 1:
            continue
        else:
            seg_feature_class = seg_feature_no_batch * class_label
            seg_feature_class_center = torch.sum(seg_feature_class, dim=0, keepdim=True) / class_num
            fg_pixel_dis = 1-compute_cos(seg_feature_no_batch, seg_feature_class_center)
            pixel_dis += (torch.sum(fg_pixel_dis*class_label,dim=0)/ class_num)
            fg_center_num += 1
            if fg_center_num == 1:
                seg_feature_fg_center = seg_feature_class_center
            else:
                seg_feature_fg_center = torch.cat([seg_feature_fg_center, seg_feature_class_center], dim=0)

    pixel_dis = pixel_dis / (fg_center_num+bg_center_num)

    if batch_num >= 1 and torch.sum(bg_num) >= 1:

        fg_fg_cos = 1 + compute_cos(seg_feature_fg_center, seg_feature_fg_center)
        fg_bg_cos = 1 + compute_cos(seg_feature_fg_center, seg_feature_bg_center)

        fg_fg_cos = fg_fg_cos - torch.diag(torch.diag(fg_fg_cos))
        if fg_fg_cos.shape[0]>1:
            fg_fg_loss = torch.sum(fg_fg_cos) / (fg_fg_cos.shape[0] * (fg_fg_cos.shape[1] - 1))

        else:
            fg_fg_loss = torch.zeros([1]).cuda()
        fg_bg_loss = torch.sum(fg_bg_cos) / (fg_bg_cos.shape[0] * fg_bg_cos.shape[1])
        dis_loss = 0.5 * fg_fg_loss.cuda() + 0.5 * fg_bg_loss.cuda()

    elif torch.sum(bg_num) < 1:
        fg_norm2 = torch.norm(seg_feature_fg_center, 2, 1).view(-1, 1)

        fg_fg_cos = 1 + torch.div(torch.mm(seg_feature_fg_center, seg_feature_fg_center.t()),
                                  torch.mm(fg_norm2, fg_norm2.t()) + 1e-7)

        fg_fg_cos = fg_fg_cos - torch.diag(torch.diag(fg_fg_cos))

        if fg_fg_cos.shape[0]>1:
            fg_fg_loss = torch.sum(fg_fg_cos) / (fg_fg_cos.shape[0] * (fg_fg_cos.shape[1] - 1))

        else:
            fg_fg_loss = torch.zeros([1]).cuda()

        dis_loss = 0.5 * fg_fg_loss + 1

    else:
        dis_loss = torch.zeros([1]).cuda()

    return dis_loss.cuda()+pixel_dis.cuda()