import cv2
import numpy as np
import argparse
import torch
import os

from sam2.build_sam import build_sam2  # MedSAM v2 的模型构建函数
from sam2.sam2_image_predictor import SAM2ImagePredictor  # MedSAM v2 的预测器类

def medsam_point(imagepath, maskpath, quanzhong):
    """
    参数：
        imagepath (str): 原始 RGB 图像路径
        maskpath (str): 二值 mask 路径，用于生成提示点
        quanzhong (str): MedSAM v2 模型权重路径（ckpt）
    返回：
        tezheng_dict (dict): 多尺度特征字典，例如 {"scale1": Tensor, "scale2": Tensor, ...}
    """
    # 构造命令行参数解析器（用于兼容旧接口）
    parser = argparse.ArgumentParser(description="run inference on testing set based on MedSAM v2")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")
    parser.add_argument(
        "--config", type=str,
        default="configs/sam2.1_hiera_t512.yaml",
        help="模型结构配置文件路径（yaml）"
    )
    args, _ = parser.parse_known_args()

    # Step 1: 读取图像并转为 RGB 格式，转换为灰度图像
    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: 构建 MedSAM v2 模型并加载权重
    model = build_sam2(
        config_file=args.config,
        ckpt_path=quanzhong,
        device=args.device,
    )
    # 修改代码-----------------------------------------------------------------------------------------------
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)
    tezheng_dict = predictor.get_multiscale_embeddings()
    return tezheng_dict




