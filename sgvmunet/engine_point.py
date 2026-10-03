import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix, f1_score
from utils import save_imgs,save_prediction
import sys
sys.path.append("/tmp/mamba+sam/MedSAM2-main")
# from medsam import *
from medsam2_point import *






import torch.distributed as dist
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def gather_numpy_array(array: np.ndarray) -> np.ndarray:
    """在分布式环境下收集所有 GPU 的 numpy array"""
    if not dist.is_initialized():
        return array  # 非分布式直接返回
    tensor = torch.tensor(array, device="cuda")
    tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor)
    concat = torch.cat(tensors_gather, dim=0)
    return concat.cpu().numpy()


def random_mask(x, mask_ratio=0.75):
    # x: [B, C, H, W]
    mask = (torch.rand(x.size(0), 1, x.size(2), x.size(3), device=x.device) > mask_ratio).float()
    x_masked = x * mask
    return x_masked, mask


# 第三阶段训练-----------------------------------------------------------------------------------------------------
def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    """
    train model for one epoch
    """
    model.train()
    loss_list = []

    feature_keys = ["feat256_96", "feat128_192", "feat64_384", "feat64_768"]
    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets, features = data
        images = images.cuda(non_blocking=True).float()
        images = (images - images.mean(dim=(1, 2, 3), keepdim=True)) / \
                 (images.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
        targets = targets.cuda(non_blocking=True).float()
        # 将 features 从 list 转成 dict
        features = {key: feat for key, feat in zip(feature_keys, features)}
        # 传递给模型的 forward 函数时使用 external_feats
        # print(
        #     "images:", images.min().item(), images.max().item(), images.dtype,
        # )
        # for k, v in features.items():
        #     print(k, v.min().item(), v.max().item(), v.dtype)

        out = model(images, features)
        loss = criterion(out, targets)
        # ✅ 反向传播
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if is_main_process():
            writer.add_scalar('loss', loss, global_step=step)
            if iter % config.print_interval == 0:
                log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
                print(log_info)
                logger.info(log_info)

    scheduler.step()
    return step, np.mean(loss_list)
def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    model.eval()
    preds, gts, loss_list = [], [], []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk, features = data
            img = img.cuda(non_blocking=True).float()
                      img = (img - img.mean(dim=(1, 2, 3), keepdim=True)) / \
                     (img.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
            msk = msk.cuda(non_blocking=True).float()

            # ✅ 多尺度特征处理
            feature_keys = ["feat256_96", "feat128_192", "feat64_384", "feat64_768"]
            if isinstance(features, (list, tuple)):
                features = {key: feat.cuda(non_blocking=True).float()
                            for key, feat in zip(feature_keys, features)}
            else:
                # 如果是 dict 的话，也强制放入设备
                features = {k: v.cuda(non_blocking=True).float()
                            for k, v in features.items()}

            out = model(img, features)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            gts.append(msk.squeeze(1).cpu().numpy())
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.squeeze(1).cpu().numpy())

    # === 聚合所有 GPU 的结果 ===
    preds = gather_numpy_array(np.array(preds).reshape(-1))
    gts   = gather_numpy_array(np.array(gts).reshape(-1))
    loss_list = gather_numpy_array(np.array(loss_list))

    if is_main_process():
        if epoch % config.val_interval == 0:
            y_pre = np.where(preds >= config.threshold, 1, 0)
            y_true = np.where(gts >= 0.5, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

            accuracy = float(TN + TP) / np.sum(confusion) if np.sum(confusion) != 0 else 0
            sensitivity = float(TP) / (TP + FN) if (TP + FN) != 0 else 0
            specificity = float(TN) / (TN + FP) if (TN + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
            miou = float(TP) / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

            log_info = (
                f"val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, "
                f"f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, "
                f"specificity: {specificity}, sensitivity: {sensitivity}, "
                f"confusion_matrix: {confusion}"
            )
            print(log_info)
            logger.info(log_info)
        else:
            log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
            print(log_info)
            logger.info(log_info)

    return np.mean(loss_list)

def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    model.eval()
    preds, gts, loss_list = [], [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk, features = data
            img = img.cuda(non_blocking=True).float()
                       img = (img - img.mean(dim=(1, 2, 3), keepdim=True)) / \
                     (img.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
            msk = msk.cuda(non_blocking=True).float()

            # ✅ 多尺度特征处理
            feature_keys = ["feat256_96", "feat128_192", "feat64_384", "feat64_768"]
            if isinstance(features, (list, tuple)):
                features = {key: feat.cuda(non_blocking=True).float()
                            for key, feat in zip(feature_keys, features)}
            else:
                features = {k: v.cuda(non_blocking=True).float()
                            for k, v in features.items()}

            out = model(img, features)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            gts.append(msk.squeeze(1).cpu().numpy())
            if isinstance(out, tuple):
                out = out[0]
            preds.append(out.squeeze(1).cpu().numpy())

            # ✅ 保存图片
            save_imgs(img, msk, out, i,
                      config.work_dir + 'outputs/',
                      config.datasets,
                      config.threshold,
                      test_data_name=test_data_name)

    # ✅ 本地计算，不 gather
    preds = np.array(preds).reshape(-1)
    gts   = np.array(gts).reshape(-1)
    loss_list = np.array(loss_list)

    y_pre = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

    accuracy = float(TN + TP) / np.sum(confusion) if np.sum(confusion) != 0 else 0
    sensitivity = float(TP) / (TP + FN) if (TP + FN) != 0 else 0
    specificity = float(TN) / (TN + FP) if (TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / (TP + FP + FN) if (TP + FP + FN) != 0 else 0

    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)

    log_info = (
        f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou}, '
        f'f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
        f'specificity: {specificity}, sensitivity: {sensitivity}, '
        f'confusion_matrix: {confusion}'
    )
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list)












def test_one_epoch_point1(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            # if i % config.save_interval == 0:
            #     # 调用函数示例
            #     # save_prediction(out, i, '/tmp/pycharm_project_859/VM-UNet/')
            #     save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
            save_prediction(out, i, '/tmp/mamba+sam/VM-UNet/train_raw_mask')
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch_point2(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            # if i % config.save_interval == 0:
            #     # 调用函数示例
            #     # save_prediction(out, i, '/tmp/pycharm_project_859/VM-UNet/')
            #     save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
            save_prediction(out, i, '/tmp/mamba+sam/VM-UNet/val_raw_mask/')
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)