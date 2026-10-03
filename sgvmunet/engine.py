import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs, save_prediction
import sys
import shutil
import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append("/tmp/pycharm_project_859/MedSAM-main")
# from medsam import *
# from medsam_point import *


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
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            # feature = torch.randn(1, 256, 64, 64)

            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
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
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            # feature = torch.randn(1, 256, 64, 64)

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
            save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                      test_data_name=test_data_name)
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

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
            # img, msk = data
            # 修改后-------------------------------------------------------------------------------------------------------------------
            img = data[0]  # 假设图像是第一个元素
            msk = data[1]  # 假设mask是第二个元素
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
            save_prediction(out, i, f'{path}/train_raw_mask/')
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

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
            # img, msk = data
            # 修改后-------------------------------------------------------------------------------------------------------------------
            img = data[0]  # 假设图像是第一个元素
            msk = data[1]  # 假设mask是第二个元素
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
            save_prediction(out, i, f'{path}/val_raw_mask/')
            #save_prediction(out, i, '/tmp/just_for_try/VM-UNet/val_raw_mask/')
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

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

def test_one_epoch_point3(test_loader,
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
            #save_prediction(out, i, '/tmp/mamba+sam/VM-UNet/train_raw_mask/')
            #save_prediction(out, i, '/tmp/just_for_try/VM-UNet/train_raw_mask/')
            save_prediction(out, i, f'{path}/predict_one/')
            # save_imgs(img, msk, out, i, '/tmp/just_for_try/VM-UNet/train_raw_mask/', config.datasets, config.threshold,
            #           test_data_name=test_data_name)
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

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
def test_one_epoch_point4(test_loader,
                          model,
                          criterion,
                          logger,
                          config,
                          test_data_name=None):
    import os
    from utils import visualize_prediction  # 确保你 utils.py 里有这个函数
    os.makedirs("results/val_vis", exist_ok=True)

    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk, filename = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            # 预测概率图
            prob_map = torch.sigmoid(out).squeeze(1)  # (B, H, W)

            # 二值化
            binary_mask = (prob_map >= config.threshold).float()

            # 保存三种格式
            image_np = img[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
            gt_mask_np = msk[0].squeeze().cpu().numpy()
            prob_map_np = prob_map[0].cpu().numpy()  # 连续概率图
            # print(f"预测概率范围：min={prob_map_np.min()}, max={prob_map_np.max()}, mean={prob_map_np.mean()}")
            binary_mask_np = binary_mask[0].cpu().numpy()

            vis_filename = os.path.basename(filename[0])

            # 多阈值效果保存
            # for th in [0.1, 0.2, 0.3, 0.4, 0.5]:
            #     temp_mask = (prob_map_np >= th).astype(np.uint8) * 255
            #     cv2.imwrite(f"results/val_vis/{vis_filename}_binary_th{int(th * 100)}.png", temp_mask)

            # 最终组合可视化
            visualize_prediction(
                image=image_np,
                pred_mask=prob_map_np,
                gt_mask=gt_mask_np,
                save_path=f"results/val_vis/{vis_filename}_vis.png",
                apply_threshold=False,  # 是否二值化
                threshold=0.3
            )

            # 评估逻辑保持不变
            msk_np = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk_np)
            preds.append(prob_map.cpu().detach().numpy())
            save_prediction(prob_map.cpu().detach().numpy(), i, f'{path}/val_raw_mask/')

        # -------- 新增混淆矩阵计算部分 -------- #
    preds_np = np.concatenate(preds, axis=0)  # (N, H, W)
    gts_np = np.concatenate(gts, axis=0)  # (N, H, W)

    y_pred = np.where(preds_np >= config.threshold, 1, 0).astype(np.uint8).flatten()
    y_true = np.where(gts_np >= 0.5, 1, 0).astype(np.uint8).flatten()

    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = confusion.ravel()

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)

    log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, ' \
               f'accuracy: {accuracy:.4f}, specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, ' \
               f'confusion_matrix: {confusion.tolist()}'
    print(log_info)
    print(f"当前使用的阈值：{config.threshold}")
    logger.info(log_info)

    return np.mean(loss_list)


import matplotlib.pyplot as plt
def save_feature_map(feat, feat_name, save_path):

    feat = feat.detach().cpu()[0]

    vis = torch.norm(
        feat,
        p=2,
        dim=0
    )

    vis = vis.numpy()

    vis = (
        vis - vis.min()
    ) / (
        vis.max() - vis.min() + 1e-8
    )

    plt.figure(figsize=(6,6))
    plt.imshow(vis, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

from sklearn.decomposition import PCA
def save_patch_mosaic(
        patches,
        meta,
        save_path,
        gap=4
):
    """
    patches:
        [N,H,W,C]

    按切分顺序拼回去
    """
    patches = patches.cpu()
    N, H, W, C = patches.shape
    per_info = meta["per_sample_info"][0]
    idxes = per_info["idxes"]
    rows = sorted(
        list(
            set(
                idx["i"]
                for idx in idxes
            )
        )
    )
    cols = sorted(
        list(
            set(
                idx["j"]
                for idx in idxes
            )
        )
    )
    n_row = len(rows)
    n_col = len(cols)
    canvas_h = n_row * H + (n_row - 1) * gap
    canvas_w = n_col * W + (n_col - 1) * gap
    canvas = torch.zeros(
        canvas_h,
        canvas_w,
        3
    )
    for p in range(N):
        patch = patches[p]
        ################################################
        # PCA RGB
        ################################################
        feat = patch.reshape(-1, C).numpy()
        rgb = PCA(
            n_components=3
        ).fit_transform(feat)
        rgb = rgb.reshape(H, W, 3)
        rgb = (
            rgb - rgb.min()
        ) / (
            rgb.max() - rgb.min() + 1e-8
        )
        i = idxes[p]["i"]
        j = idxes[p]["j"]
        row_idx = rows.index(i)
        col_idx = cols.index(j)
        top = row_idx * (H + gap)
        left = col_idx * (W + gap)
        canvas[
            top:top+H,
            left:left+W
        ] = torch.tensor(rgb)
    plt.figure(
        figsize=(12,8)
    )
    plt.imshow(canvas.numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()


from skimage.morphology import skeletonize
from medpy.metric.binary import hd95
def compute_cldice(pred, gt):
    """
    pred, gt: [H,W] uint8
    """
    pred_skel = skeletonize(pred > 0)
    gt_skel = skeletonize(gt > 0)
    tprec = (np.logical_and(pred_skel, gt).sum()/(pred_skel.sum() + 1e-8))
    tsens = (np.logical_and(gt_skel, pred).sum()/(gt_skel.sum() + 1e-8))
    cldice = (2 * tprec * tsens /(tprec + tsens + 1e-8) )
    return cldice

def test_one_epoch_test(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    model.eval()
    preds, gts, loss_list = [], [], []
    preds, gts, loss_list = [], [], []
    all_pred_masks = []
    all_gt_masks = []
    feature_dir = os.path.join(
        config.work_dir,
        "feature_vis"
    )
    os.makedirs(
        feature_dir,
        exist_ok=True
    )
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
                      sample_dir = os.path.join(
                feature_dir,
                f"image_{i:04d}"
            )
            os.makedirs(
                sample_dir,
                exist_ok=True
            )
            for name, module in model.named_modules():
                if not hasattr(module, "debug_feats"):
                    continue
                             module_dir = os.path.join(
                    sample_dir,
                    name.replace(".", "_")
                )
                os.makedirs(
                    module_dir,
                    exist_ok=True
                )
             
                if (
                        "patches" in module.debug_feats
                        and
                        "patch_meta" in module.debug_feats
                ):
                    save_patch_mosaic(
                        module.debug_feats["patches"],
                        module.debug_feats["patch_meta"],
                        os.path.join(
                            module_dir,
                            "patch_mosaic.png"
                        ),
                        gap=4
                    )

                for feat_name, feat in module.debug_feats.items():

                    if feat_name in [
                        "patches",
                        "patch_meta"
                    ]:
                        continue

                    save_path = os.path.join(
                        module_dir,
                        f"{feat_name}.png"
                    )

                    save_feature_map(
                        feat,
                        feat_name,
                        save_path
                    )

            loss = criterion(out, msk)
            loss_list.append(loss.item())

            # gts.append(msk.squeeze(1).cpu().numpy())
            # if isinstance(out, tuple):
            #     out = out[0]
            # preds.append(out.squeeze(1).cpu().numpy())
            if isinstance(out, tuple):
                out = out[0]
            pred_np = out.squeeze(1).cpu().numpy()
            gt_np = msk.squeeze(1).cpu().numpy()
            preds.append(pred_np)
            gts.append(gt_np)
            pred_bin = (pred_np >= config.threshold).astype(np.uint8)
            gt_bin = (gt_np >= 0.5).astype(np.uint8)
            for b in range(pred_bin.shape[0]):
                all_pred_masks.append(pred_bin[b])
                all_gt_masks.append(gt_bin[b])

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
    cldice_scores = []
    hd95_scores = []

    for pred_mask, gt_mask in zip(all_pred_masks,all_gt_masks):
        # clDice
        cldice_scores.append(compute_cldice(pred_mask,gt_mask))
        # HD95
        if pred_mask.sum() > 0 and gt_mask.sum() > 0:
            try:
                hd95_scores.append(hd95(pred_mask.astype(bool),gt_mask.astype(bool)))
            except:
                pass
    mean_cldice = (
        np.mean(cldice_scores)
        if len(cldice_scores) > 0
        else 0
    )
    mean_hd95 = (
        np.mean(hd95_scores)
        if len(hd95_scores) > 0
        else 0
    )

    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)

    # log_info = (
    #     f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou}, '
    #     f'f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, '
    #     f'specificity: {specificity}, sensitivity: {sensitivity}, '
    #     f'confusion_matrix: {confusion}'
    # )
    log_info = (
        f'test of best model, '
        f'loss: {np.mean(loss_list):.4f}, '
        f'miou: {miou:.4f}, '
        f'f1_or_dsc: {f1_or_dsc:.4f}, '
        f'clDice: {mean_cldice:.4f}, '
        f'HD95: {mean_hd95:.4f}, '
        f'accuracy: {accuracy:.4f}, '
        f'specificity: {specificity:.4f}, '
        f'sensitivity: {sensitivity:.4f}, '
        f'confusion_matrix: {confusion}'
    )
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list)




def test_one_epoch_point5(test_loader,
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

            img, msk, filename = data

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
            save_prediction(out, i, f'{path}/train_raw_mask/')
            # save_imgs(img, msk, out, i, f'{path}/train_raw_mask/', config.datasets, config.threshold,
            #           test_data_name=test_data_name)
            #save_prediction(out, i, '/tmp/just_for_try/VM-UNet/val_raw_mask/')
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, ' \
               f'accuracy: {accuracy:.4f}, specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, ' \
               f'confusion_matrix: {confusion.tolist()}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)