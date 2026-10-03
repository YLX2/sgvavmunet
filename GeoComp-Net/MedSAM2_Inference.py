
import cv2
import torch
import numpy as np

from .sam2.build_sam import build_sam2
from .sam2.sam2_image_predictor import SAM2ImagePredictor

def MedSAM_Inference(imagepath, maskpath, quanzhong):
    """
    使用 MedSAM v2 对图像进行分割特征提取，基于点提示。

    Args:
        imagepath (str): 输入图像路径（RGB）
        maskpath (str): 掩膜图像路径（用于采样前景点）
        quanzhong (str): MedSAM 权重路径（.pth）

    Returns:
        torch.Tensor: 提取的图像特征向量
    """

    # Step 1: 加载图像并转换为 RGB
    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 2: 读取掩膜并获取前景坐标点（白色区域）
    mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    y_coords, x_coords = np.where(mask == 255)
    points = np.array(list(zip(x_coords, y_coords)))

    # Step 3: 采样点提示（等间隔取10个点）
    total_points = len(points)
    if total_points < 10:
        raise ValueError("掩膜区域过小，无法采样足够点。")
    skip_interval = total_points // 10
    input_point = points[::skip_interval][:10]  # 限制最多10个点
    input_label = np.ones(len(input_point), dtype=np.int32)

    # Step 4: 构建模型并加载权重
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_sam2(checkpoint=quanzhong, model_type="vit_b")  # 默认是 vit_b
    model.to(device)
    model.eval()

    # Step 5: 创建预测器并加载图像
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image, image_format="RGB", features_only=True)

    # Step 6: 点提示预测掩码（非必须）
    _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    # Step 7: 获取图像特征向量
    tezhengxiangliang = predictor.get_image_embedding()  # torch.Tensor

    return tezhengxiangliang
