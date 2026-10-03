import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 或其他你想要的 GPU 编号

from torch.utils.data import DataLoader
import time
import sys

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_directory = os.path.dirname(current_file_path)
# 获取上一级目录
parent_directory = os.path.dirname(current_directory)

sys.path.append(f"{parent_directory}/MedSAM2-main")
from datasets.dataset_point import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet_point
from train_point import train_point
from engine_point import *
# from medsam import *
from medsam2_point import *
from utils import *
from configs.config_setting import setting_config,valsetting_config,setting_config_point

import shutil
import warnings
warnings.filterwarnings("ignore")

def main(config,config0):
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")

    total_start_time = time.time()  # 记录总训练开始时间

    # 检查是否需要重新生成mask

    need_generate_mask = not check_existing_masks(train_mask_dir, val_mask_dir)

    if need_generate_mask:
        clear_folder(train_mask_dir)  # 清空输出文件夹
        clear_folder(val_mask_dir)
        train_point(config1)  # 生成mask
    else:
        print("检测到已存在的mask文件，跳过mask生成步骤")

    # ✅ 检查.pt特征文件
    if not check_existing_features(train_output_dir, val_output_dir):
        print("未找到.pt特征文件，开始生成...")
        clear_folder(train_output_dir)  # 清空输出文件夹
        clear_folder(val_output_dir)
        # 处理训练数据
        process_images(train_mask_dir, train_image_dir, train_output_dir, model_path)
        # 处理验证数据
        process_images(val_mask_dir, val_image_dir, val_output_dir, model_path)
    else:
        print("检测到已有.pt特征文件，跳过特征生成步骤")


    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    # resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    resume_model = config.model_config['resume_ckpt']
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#') # 512*512的原图整体缩放myResize到256*256
    train_dataset = NPY_datasets(config.data_path, train_output_dir, config, train=True) # torch.Size([256, 64, 64])# 即 myResize(256, 256)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, val_output_dir, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet_point(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'] #---------------------------------------------------------------------
            # load_ckpt_path=None
        )
        model.load_from()

    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops_point(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    if resume_model is not None and os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    train_losses = []
    val_losses = []
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step,train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )
        train_losses.append(train_loss)

        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config0
        )
        val_losses.append(loss)

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )
        total_end_time = time.time()  # 记录总训练结束时间
        total_training_time = total_end_time - total_start_time
        logger.info(f"Total training time: {total_training_time:.2f} seconds.")
    return train_losses, val_losses


def clear_folder(folder_path):
    """清空文件夹并重新创建同名空文件夹。"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def process_images(mask_dir, image_dir, output_dir, model_path):
    """处理图像和掩码文件，保存处理结果。"""
    mask_files = sorted(os.listdir(mask_dir))
    image_files = sorted(os.listdir(image_dir))
    for i, (mask_file, image_file) in enumerate(zip(mask_files, image_files)):
        mask_path = os.path.join(mask_dir, mask_file)
        image_path = os.path.join(image_dir, image_file)
        medsam_result = medsam_point(image_path, mask_path, model_path)
        # torch.save(medsam_result, os.path.join(output_dir, f"{i}.pt"))
        torch.save({
            "feat256_96": medsam_result["feat256_96"].cpu(),
            "feat128_192": medsam_result["feat128_192"].cpu(),
            "feat64_384": medsam_result["feat64_384"].cpu(),
            "feat64_768": medsam_result["feat64_768"].cpu(),
        }, os.path.join(output_dir, f"{i}.pt"))
        print(f"保存{i}.pt文件完成")


def check_existing_masks(train_mask_dir, val_mask_dir):
    """
    检查训练和验证的特征文件是否已存在
    """
    # 检查训练目录
    if not os.path.exists(train_mask_dir) or not os.listdir(train_mask_dir):
        print(f"训练特征目录 {train_mask_dir} 不存在或为空")
        return False
    # 检查验证目录
    if not os.path.exists(val_mask_dir) or not os.listdir(val_mask_dir):
        print(f"验证特征目录 {val_mask_dir} 不存在或为空")
        return False
    # 检查文件是否为有效的.png文件
    train_files = os.listdir(train_mask_dir)
    val_files = os.listdir(val_mask_dir)
    # 简单检查：确保目录中至少有一些.png文件
    train_png_files = [f for f in train_files if f.lower().endswith('.png')]
    val_png_files = [f for f in val_files if f.lower().endswith('.png')]
    if len(train_png_files) == 0 or len(val_png_files) == 0:
        print("未找到有效的.png mask文件")
        return False
    print(f"找到 {len(train_png_files)} 个训练mask文件和 {len(val_png_files)} 个验证mask文件")
    return True


def check_existing_features(train_output_dir, val_output_dir):
    """
    检查训练和验证的特征文件是否已存在
    """
    # 检查训练目录
    if not os.path.exists(train_output_dir) or not os.listdir(train_output_dir):
        print(f"训练特征目录 {train_output_dir} 不存在或为空")
        return False
    # 检查验证目录
    if not os.path.exists(val_output_dir) or not os.listdir(val_output_dir):
        print(f"验证特征目录 {val_output_dir} 不存在或为空")
        return False
    # 检查文件是否为有效的.pt文件
    train_files = os.listdir(train_output_dir)
    val_files = os.listdir(val_output_dir)
    train_pt_files = [f for f in train_files if f.lower().endswith('.pt')]
    val_pt_files = [f for f in val_files if f.lower().endswith('.pt')]
    if len(train_pt_files) == 0 or len(val_pt_files) == 0:
        print("未找到有效的.pt特征文件")
        return False
    print(f"找到 {len(train_pt_files)} 个训练特征文件和 {len(val_pt_files)} 个验证特征文件")
    return True






if __name__ == '__main__':
    print("当前文件的绝对路径:", current_file_path)
    print("当前文件所在的目录:", current_directory)
    print("上一级目录:", parent_directory)
    config = setting_config
    config0 = valsetting_config
    config1 = setting_config_point
    # 定义路径
    train_mask_dir = f'{current_directory}/train_raw_mask'
    train_image_dir = config.data_path + 'train/images'
    # train_output_dir = f"{parent_directory}/MedSAM2-main/train_tezhengxiangliang"
    train_output_dir = '/data/mydata/train_tezhengxiangliang/'
    model_path = f"{parent_directory}/MedSAM2-main/checkpoints/sam2.1_hiera_tiny.pt"

    val_mask_dir = f'{current_directory}/val_raw_mask'
    val_image_dir = config.data_path + 'val/images'
    # val_output_dir = f"{parent_directory}/MedSAM2-main/val_tezhengxiangliang"
    val_output_dir = '/data/mydata/val_tezhengxiangliang/'
    # 图片保存路径
    output_folder = current_directory
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_mask_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    # train_losses, val_losses = main(config)--------------------------------------------------------------------------------------
    train_losses , val_losses = main(config, config0)


    # 设置绘图
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)  # 生成一个epoch列表，从1开始
    plt.plot(epochs, train_losses, 'r', label='Training Loss')  # 使用红色圆点连线显示训练损失
    plt.plot(epochs, val_losses, 'b', label='Validation Loss')  # 使用蓝色圆点连线显示验证损失
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    output_file_path = os.path.join(output_folder, 'learning_curve.png')
    plt.savefig(output_file_path)  # 保存为PNG文件到指定路径




