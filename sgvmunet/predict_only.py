import os
import sys
import torch
from torch.utils.data import DataLoader
from engine import *
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
parent_directory = os.path.dirname(current_directory)
sys.path.append(f"{parent_directory}/MedSAM2-main")
from datasets.dataset_point import NPY_datasets
from models.vmunet.vmunet import VMUNet_point
from engine_point import test_one_epoch
from utils import get_logger, set_seed
from configs.config_setting import setting_config
def main(config):
    print('#========== Test Only ==========')
    # ---------- logger ----------
    log_dir = os.path.join(config.work_dir, 'test_log')
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger('test', log_dir)
    # ---------- CUDA ----------
    set_seed(config.seed)
    torch.cuda.empty_cache()
    # ---------- 数据 ----------
    val_output_dir = f"{parent_directory}/MedSAM2-main/val_tezhengxiangliang"
    val_dataset = NPY_datasets(
        config.data_path,
        val_output_dir,
        config,
        train=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True
    )
    logger.info(f"Validation samples: {len(val_dataset)}")
    # ---------- 模型 ----------
    model_cfg = config.model_config
    model = sgva_point(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None
    )
    model.vmunet.debug_shapes = True # 输出调试尺寸---------------------------------------------------------------------
    model.load_from()
    model = model.cuda()
    for m in model.modules():
        if m.__class__.__name__ == "EVSS":
            m.save_feature = True
    # ---------- 加载权重 ----------
    # best_ckpt = os.path.join(config.data_path, 'checkpoints', 'best.pth')
    assert os.path.exists(best_ckpt), f"❌ 未找到权重文件: {best_ckpt}"

    logger.info(f"Loading checkpoint: {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, map_location='cpu'), strict=False)

    # ---------- loss ----------
    criterion = config.criterion

    # ---------- 测试 ----------
    logger.info('#---------- Testing ----------#')
    test_loss = test_one_epoch_test(
        val_loader,
        model,
        criterion,
        logger,
        config
    )
    logger.info(f"✅ Test Loss: {test_loss:.6f}")
    print(f"✅ Test Loss: {test_loss:.6f}")
    return test_loss

if __name__ == '__main__':
    config = setting_config
    config.data_path = "/data/mydata/syntax_aug/"
    # config.data_path = "/home/ylx/SAM-VMNet-main/VM-UNet/data/syntax_test1/"
    main(config)
