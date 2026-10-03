import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),#原代码---------------------------------------------------------------------------------------
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )

    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'WP_CosineAnnealingLR','ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_CosineAnnealingLR':  # 🔥 关键新增------------------------------------------------------------------
        warmup = LinearLR(
            optimizer,
            start_factor=config.min_lr / config.base_lr,
            end_factor=1.0,
            total_iters=config.warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.min_lr
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_epochs]
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


# 原代码-------------------------------------------------------------------------------------------------------------------
# def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
#     img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
#     img = img / 255. if img.max() > 1.1 else img
#     if datasets == 'retinal':
#         msk = np.squeeze(msk, axis=0)
#         msk_pred = np.squeeze(msk_pred, axis=0)
#     else:
#         msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
#         msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
#
#     plt.figure(figsize=(7,15))
#
#     plt.subplot(3,1,1)
#     plt.imshow(img)
#     plt.axis('off')
#
#     plt.subplot(3,1,2)
#     plt.imshow(msk, cmap= 'gray')
#     plt.axis('off')
#
#     plt.subplot(3,1,3)
#     plt.imshow(msk_pred, cmap = 'gray')
#     plt.axis('off')
#     plt.axis('off')
#
#     if test_data_name is not None:
#         save_path = save_path + test_data_name + '_'
#     plt.savefig(save_path + str(i) +'.png')
#     plt.close()

import numpy as np
import matplotlib.pyplot as plt
# 阶段三------------------------------------------------------------------------------------------------------------
def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img

    msk = msk.detach().cpu().numpy() if hasattr(msk, "detach") else msk
    msk_pred = msk_pred.detach().cpu().numpy() if hasattr(msk_pred, "detach") else msk_pred

    if datasets == 'retinal':
        msk = np.squeeze(msk)
        msk_pred = np.squeeze(msk_pred)
    else:
        msk = np.squeeze(msk)
        msk = np.where(msk > 0.5, 1, 0)
        msk_pred = np.squeeze(msk_pred)
        msk_pred = np.where(msk_pred > threshold, 1, 0)

    # Debug 打印
    # print(f"[DEBUG] img shape: {img.shape}, msk shape: {msk.shape}, msk_pred shape: {msk_pred.shape}")

    plt.figure(figsize=(7, 15))
    plt.subplot(3, 1, 1); plt.imshow(img); plt.axis('off')
    plt.subplot(3, 1, 2); plt.imshow(msk, cmap='gray'); plt.axis('off')
    plt.subplot(3, 1, 3); plt.imshow(msk_pred, cmap='gray'); plt.axis('off')

    # 拼接保存路径
    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存
    plt.savefig(save_path + str(i) + '.png')
    plt.close()






def save_prediction(msk_pred, i, save_path, threshold=0.5):
    # 应用阈值，生成二值化预测掩码
    msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    # 使用matplotlib创建图像
    plt.figure(figsize=(6, 6))
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')  # 不显示坐标轴

    # 构建保存路径，包括文件名
    file_name = f"{save_path}prediction_{i}.png"

    # 保存预测掩码图像到指定路径，注意bbox_inches参数
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0)

    # 关闭图像以释放内存
    plt.close()




class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
    

class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight
    
    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss

# 阶段三原代码-------------------------------------------------------------------------------------------------------------------
class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd
    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        # diceloss = self.dice(torch.sigmoid(pred), target)
        loss = self.wb * bceloss + self.wd * diceloss
        return loss



#     修改后---------------------------------------------------------------------------------------------------
class newDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(newDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 保证 pred 是概率
        pred = torch.sigmoid(pred)
        target = target.float()

        intersection = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss.mean()


class valBceDiceLoss(nn.Module):
    def __init__(self, wb=0.5, wd=1.5):
        """
        wb: BCE 权重
        wd: Dice 权重
        """
        super(valBceDiceLoss, self).__init__()
        # 用 BCEWithLogitsLoss 更稳定（模型输出不需要先 sigmoid）
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = newDiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.wb * bce_loss + self.wd * dice_loss


class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss



class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask

class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask

# 原代码没有除255
class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
    # 阶段三原代码
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized))
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk

from thop import profile		 ## 导入thop模块
# 无多尺度注入前代码--------------------------------------------------------------------------------------------------------
# def cal_params_flops_point(model, size, logger):
#     input = torch.randn(1, 3, size, size).cuda()
#     feature = torch.randn(1, 256, 64, 64)
#     # 确定设备
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # 将模型和输入张量移动到同一设备上
#     model = model.to(device)
#     input = input.to(device)
#     feature = feature.to(device)
#     flops, params = profile(model, inputs=(input,feature))
#     print('flops',flops/1e9)			## 打印计算量
#     print('params',params/1e6)			## 打印参数量
#
#     total = sum(p.numel() for p in model.parameters())
#     print("Total params: %.2fM" % (total/1e6))
#     logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')
#
def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    # 确定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将模型和输入张量移动到同一设备上
    model = model.to(device)
    input = input.to(device)
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')
def cal_params_flops_point(model, size, logger):
    """
    计算模型参数量和 FLOPs（适配支持 external_feats 的模型）
    """
    # 输入张量
    input = torch.randn(1, 3, size, size).cuda()

    # 模拟 MedSAM2 输出的多尺度特征字典（与 forward_features_up 对应）
    feature = {
        "feat256_96": torch.randn(1, 96, 256, 256).cuda(),
        "feat128_192": torch.randn(1, 192, 128, 128).cuda(),
        "feat64_384": torch.randn(1, 384, 64, 64).cuda(),
        "feat64_768": torch.randn(1, 768, 64, 64).cuda(),
    }

    # 确定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 将模型移动到设备上
    model = model.to(device)

    # 手动预跑一次，确保 final_up 和 final_conv 都固定好形状
    with torch.no_grad():
        model.vmunet(input, feature)

    # thop 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(input, feature))

    print('flops', flops / 1e9)      # 打印计算量（GFLOPs）
    print('params', params / 1e6)    # 打印参数量（M）
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    logger.info(
        f'flops: {flops / 1e9:.3f} G, params: {params / 1e6:.3f} M, Total params: {total / 1e6:.3f} M'
    )





def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], 
                    test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
        # cv2.imwrite(test_save_path + '/'+case + '.png', prediction*255)
    return metric_list

# ！！！新增代码（预测的可视化）连续概率图可视化----------------------------------------------------------------------------
# def visualize_prediction(image, pred_mask, gt_mask=None, save_path=None, apply_threshold=True):
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     plt.figure(figsize=(12, 4))
#
#     plt.subplot(1, 3, 1)
#     plt.title("Original Image")
#     plt.imshow(image.transpose(1, 2, 0).astype(np.uint8))
#     plt.axis("off")
#
#     plt.subplot(1, 3, 2)
#     plt.title("Predicted Mask")
#     if apply_threshold:
#         pred_display = (pred_mask >= 0.5).astype(np.uint8) * 255
#     else:
#         pred_display = pred_mask * 255
#     plt.imshow(pred_display, cmap='jet')
#     plt.axis("off")
#
#     plt.subplot(1, 3, 3)
#     plt.title("Ground Truth Mask")
#     plt.imshow(gt_mask.squeeze() * 255, cmap='gray')
#     plt.axis("off")
#
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#     plt.close()

# ！！！新增代码，预测二值掩码的可视化----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def visualize_prediction(image, pred_mask, gt_mask=None, save_path=None, apply_threshold=True, threshold=0.3):
    """
    可视化原图、预测掩码、真实掩码的组合图，支持二值化控制。
    :param image: 原始图像，(H, W, 3) 格式，numpy数组
    :param pred_mask: 预测掩码，(H, W) 格式，连续值0-1
    :param gt_mask: 真实掩码，(H, W) 格式，二值0/1
    :param save_path: 保存路径，含完整文件名
    :param apply_threshold: 是否对预测掩码二值化
    :param threshold: 二值化阈值
    """
    if apply_threshold:
        pred_display = (pred_mask >= threshold).astype(np.uint8) * 255
        title_pred = f'Pred Mask (th={threshold})'
    else:
        pred_display = (pred_mask * 255).astype(np.uint8)
        title_pred = 'Pred Prob Map'

    # 保证原图范围正确
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # 绘制
    fig, axs = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(15, 5))

    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(pred_display, cmap='gray')
    axs[1].set_title(title_pred)
    axs[1].axis('off')

    if gt_mask is not None:
        axs[2].imshow((gt_mask * 255).astype(np.uint8), cmap='gray')
        axs[2].set_title('Ground Truth Mask')
        axs[2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    else:
        plt.show()




