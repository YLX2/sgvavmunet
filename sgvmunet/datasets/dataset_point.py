from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import sys
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件所在的目录
current_directory = os.path.dirname(current_file_path)

# 获取上一级目录
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
print("当前文件的绝对路径:", current_file_path)
print("当前文件所在的目录:", current_directory)
print("上一级目录:", parent_directory)
print("上上级目录:", grandparent_directory)
sys.path.append(f"{grandparent_directory}/MedSAM2-main")
# from medsam import *
from medsam2_point import *
import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image


class NPY_datasets(Dataset):
    def __init__(self, path_Data, path_Features, config, train=True):
        super(NPY_datasets, self)
        if train:
            images_list = sorted(os.listdir(path_Data + 'train/images/'))
            masks_list = sorted(os.listdir(path_Data + 'train/masks/'))
            features_dir = '/data/mydata/train_tezhengxiangliang/'
            features_list = sorted(os.listdir(features_dir))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                msk_path = path_Data + 'train/masks/' + masks_list[i]
                feature_path = os.path.join(path_Features, features_dir, features_list[i])
                self.data.append((img_path, msk_path, feature_path))

            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data + 'val/images/'))
            masks_list = sorted(os.listdir(path_Data + 'val/masks/'))
            features_dir = '/data/mydata/val_tezhengxiangliang/'
            features_list = sorted(os.listdir(features_dir))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                feature_path = os.path.join(path_Features, features_dir, features_list[i])
                self.data.append((img_path, mask_path, feature_path))
            self.transformer = config.test_transformer
    def __getitem__(self, indx):
        img_path, msk_path, feature_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255

        feature_data = torch.load(feature_path, map_location='cpu')

        if isinstance(feature_data, dict):

            feats = []
            for key in ["feat256_96", "feat128_192", "feat64_384", "feat64_768"]:
                if key in feature_data:
                    feats.append(feature_data[key].squeeze(0))  # 去掉 batch 维度
                else:
                    raise KeyError(f"Expected key '{key}' not found in {feature_path}")
        else:

            feats = [feature_data.squeeze(0)] * 4  # 复制四份保证维度一致
        if self.transformer is not None:
            img, msk = self.transformer((img, msk))  # 图像增强
            feats = [f.to('cpu') for f in feats]  # 外部特征不变换
        return img, msk, feats
    def __len__(self):
        return len(self.data)




def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

