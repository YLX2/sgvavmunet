import h5py
import numpy as np
from PIL import Image
import os


def convert_h5_to_images(h5_file_path, output_dir):
    """
    将HDF5文件转换回原始PNG图像

    参数:
        h5_file_path: HDF5文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取HDF5文件
    with h5py.File(h5_file_path, 'r') as f:
        # 根据你的原始代码，数据集名称为"image"
        data = f['image'][:]
        print(f"数据集形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数据范围: [{np.min(data):.6f}, {np.max(data):.6f}]")

    # 数据形状是 (30, 1, 512, 512) - 4D数据
    num_images, channels, height, width = data.shape

    print(f"图像数量: {num_images}")
    print(f"通道数: {channels}")
    print(f"图像尺寸: {height}x{width}")

    # 转换每一张图像
    for img_idx in range(num_images):
        # 提取单张图像数据 (去掉通道维度)
        img_data = data[img_idx, 0, :, :]  # 取第一个通道

        # 由于数据范围是[0,255]，直接转换为uint8
        if data.dtype == np.float64:
            # 如果数据是float64类型，需要转换
            img_data_uint8 = img_data.astype(np.uint8)
        else:
            img_data_uint8 = img_data

        # 创建PIL图像并保存
        img = Image.fromarray(img_data_uint8)

        # 生成输出文件名
        output_filename = f"ori_{img_idx + 1}.png"
        output_path = os.path.join(output_dir, output_filename)

        img.save(output_path)

        print(f"已保存: {output_path} - 数值范围: [{np.min(img_data_uint8)}, {np.max(img_data_uint8)}]")


def inspect_h5_structure(h5_file_path):
    """查看HDF5文件结构"""
    print("=== HDF5文件结构 ===")
    with h5py.File(h5_file_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"数据集: {name}")
                print(f"  形状: {obj.shape}")
                print(f"  数据类型: {obj.dtype}")
                print(f"  数值范围: [{np.min(obj):.6f}, {np.max(obj):.6f}]")

        f.visititems(print_structure)



# 使用示例
if __name__ == "__main__":
    h5_file = "/data/test1_gt/test1_gt.hdf5"  # 你的HDF5文件路径
    output_directory = "/data/test1_gt"  # 输出目录

    # 首先查看文件结构
    inspect_h5_structure(h5_file)

    # 然后转换文件
    convert_h5_to_images(h5_file, output_directory)

    print("转换完成！")
