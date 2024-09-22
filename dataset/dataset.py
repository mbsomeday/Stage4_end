import numpy as np
from PIL import Image
from torchvision import transforms
import os, random
from torch.utils.data import Dataset, DataLoader
import argparse
import sys, os

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from cv_models import VARS_LOCAL, VARS_CLOUD


class dsCls_Dataset(Dataset):
    '''
        读取数据用于数据及分类
    '''
    def __init__(self, var_opt, ds_name_list, ds_label_list, txt_name):
        self.var_opt = var_opt
        self.ds_name_list = ds_name_list
        self.ds_label_list = ds_label_list
        self.txt_name = txt_name
        self.image_transformer = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.initImgLabel()

    def initImgLabel(self):
        temp_images = []
        temp_labels = []

        for idx, ds_name in enumerate(self.ds_name_list):
            ds_label = self.ds_label_list[idx]
            base_dir = self.var_opt[ds_name]
            txt_path = os.path.join(base_dir, 'dataset_txt', self.txt_name)

            # 读取 txt
            with open(txt_path, 'r') as f:
                data = f.readlines()

            for line in data:
                line = line.replace('\\', os.sep)
                line = line.strip().split()
                image_path = os.path.join(base_dir, line[0])
                temp_images.append(image_path)
                temp_labels.append(ds_label)


        # 打乱顺序
        total_num = len(temp_images)
        idx_list = list(range(total_num))
        random.shuffle(idx_list)

        images = []
        labels = []

        for idx in idx_list:
            images.append(temp_images[idx])
            labels.append(temp_labels[idx])

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label = self.labels[idx]
        label = np.array(label).astype(np.int64)
        img = Image.open(image_name)  # PIL image shape:（C, W, H）
        img = self.image_transformer(img)

        return img, label, image_name


class pedCls_Dataset(Dataset):
    '''
        读取多个数据集的数据
    '''

    def __init__(self, runOn, ds_name_list, txt_name):
        self.runOn = runOn
        self.base_dir_list = [self.runOn[ds_name] for ds_name in ds_name_list]
        self.txt_name = txt_name
        self.image_transformer = transforms.Compose([
            transforms.ToTensor()
        ])
        self.images, self.labels = self.initImgLabel()

    def initImgLabel(self):
        '''
            读取图片 和 label
        '''
        images = []
        labels = []

        for base_dir in self.base_dir_list:
            txt_path = os.path.join(base_dir, 'dataset_txt', self.txt_name)
            with open(txt_path, 'r') as f:
                data = f.readlines()

            for line in data:
                line = line.replace('\\', os.sep)
                line = line.strip().split()
                image_path = os.path.join(base_dir, line[0])
                label = line[-1]
                images.append(image_path)
                labels.append(label)

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label = self.labels[idx]
        label = np.array(label).astype(np.int64)
        img = Image.open(image_name)  # PIL image shape:（C, W, H）
        img = self.image_transformer(img)
        return img, label, image_name


class cycleGAN_Dataset(Dataset):
    '''
        返回：imgA, imgB, pathA, pathB
        把图片从A转换到B
        get_num: 若没有指定，则取两个数据集数据量的最小值
    '''
    def __init__(self, runOn, dataset_name_list, txt_name, get_num=None):
        self.dataset_dir_list = [runOn[ds_name] for ds_name in dataset_name_list]

        self.dsA_name = dataset_name_list[0]
        self.dsB_name = dataset_name_list[1]

        self.dsA_dir = self.dataset_dir_list[0]
        self.dsB_dir = self.dataset_dir_list[1]
        self.txt_name = txt_name
        self.get_num = get_num

        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.imgA, self.imgB = self.init_ImageandLabel()

        # 输出一些必要信息
        print(f'Transformation Dataset from {self.dsA_name} to {self.dsB_name}, get_num={self.get_num}')

    def init_ImageandLabel(self):
        dsA_txt_path = os.path.join(self.dsA_dir, 'dataset_txt', self.txt_name)
        with open(dsA_txt_path, 'r') as f:
            dsA_data = f.readlines()

        dsB_txt_path = os.path.join(self.dsB_dir, 'dataset_txt', self.txt_name)
        with open(dsB_txt_path, 'r') as f:
            dsB_data = f.readlines()

        dsA_num = len(dsA_data)
        dsB_num = len(dsB_data)

        if self.get_num is not None:
            if self.get_num > dsA_num or self.get_num > dsB_num:
                print(f'Dataset A total num: {dsA_num}, \ndataset B total number: {dsB_num}, \n请重新设置get_num的值!(当前get_num={self.get_num})')
        else:
            self.get_num = min(dsA_num, dsB_num)

        imgA = self.get_image(dsA_data, self.dsA_dir)
        imgB = self.get_image(dsB_data, self.dsB_dir)

        return imgA, imgB


    def get_image(self, data, base_dir):
        '''
        :param data: 读取的txt文件
        :param base_dir:
        :return: 根据读取的txt返回get_num数量的image_path
        '''

        random.shuffle(data)
        images = []

        for i in range(self.get_num):
            line = data[i]
            line = line.replace('\\', os.sep)
            line = line.strip()
            words = line.split()

            image_path = os.path.join(base_dir, words[0])
            images.append(image_path)

        return images



    def __len__(self):
        return self.get_num

    def __getitem__(self, item):
        imgA_path = self.imgA[item]
        imgB_path = self.imgB[item]

        imgA = Image.open(imgA_path)
        imgA = self.image_transformer(imgA)
        imgB = Image.open(imgB_path)
        imgB = self.image_transformer(imgB)

        return imgA, imgB









