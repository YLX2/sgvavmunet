import os
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import albumentations as A

# 路径配置
train_images_dir = "/dataimages"
train_masks_dir = "/datamasks"
save_images_dir = "/dataimages"
save_masks_dir = "/datamasks"

os.makedirs(save_images_dir, exist_ok=True)
os.makedirs(save_masks_dir, exist_ok=True)


train_transform = A.Compose([
    A.SmallestMaxSize(max_size=576),
    A.RandomCrop(height=512, width=512, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=10, p=0.7),
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.ElasticTransform(alpha=1, sigma=30, alpha_affine=5, p=0.15),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MedianBlur(blur_limit=3),
        A.GaussNoise(var_limit=(5.0, 20.0))
    ], p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,)),  # 仅对image起作用
], additional_targets={'mask': 'mask'})

# transform = A.Compose([
#     A.Rotate(limit=10, p=0.8),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.3),
#     A.RandomScale(scale_limit=0.2, p=0.7),
#     A.Affine(
#         shear={"x": (-10, 10), "y": (-10, 10)},
#         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#         rotate=(-10, 10),
#         scale=(0.8, 1.2),
#         p=0.7
#     ),
#     A.SmallestMaxSize(max_size=512),
#     A.CenterCrop(height=512, width=512),
#
#     # 温和的光度变换
#     # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
#     # A.RandomGamma(gamma_limit=(95, 105), p=0.3),
# ])

# advanced_transform = A.Compose([
#     # 更强烈的几何变换
#     A.OneOf([
#         A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
#         A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1),
#                  rotate=(-12, 12), shear=(-5, 5), p=0.7),
#     ], p=0.8),

#     A.ElasticTransform(alpha=0.8, sigma=25, alpha_affine=25, p=0.2),
#     # 更多样的光度变换
#     A.OneOf([
#         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#         A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
#         A.Blur(blur_limit=3, p=0.1),
#     ], p=0.5),

#     A.GridDistortion(num_steps=3, distort_limit=0.15, p=0.1),
# ], additional_targets={'mask': 'mask'})


num_aug_per_image = 5   # 每张增强5次
img_size = 512


image_files = sorted(os.listdir(train_images_dir))
print(f"Found {len(image_files)} training images. Augmenting {num_aug_per_image}x each...")

for img_name in tqdm(image_files):
    img_path = os.path.join(train_images_dir, img_name)
    mask_path = os.path.join(train_masks_dir, img_name)

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"⚠️ Skip invalid file {img_name} (cannot read image or mask)")
        continue

    # 确保为512大小
    image = cv2.resize(image, (img_size, img_size))
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(save_images_dir, img_name), image)
    cv2.imwrite(os.path.join(save_masks_dir, img_name), mask)

    # 生成增强版本
    base_name, ext = os.path.splitext(img_name)
    for i in range(num_aug_per_image):
        print(type(image), img_name)

        augmented = train_transform(image=image, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        aug_img_name = f"{base_name}_aug{i}{ext}"
        cv2.imwrite(os.path.join(save_images_dir, aug_img_name), aug_img)
        cv2.imwrite(os.path.join(save_masks_dir, aug_img_name), aug_mask)


