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
        print('数据集：', txt_name)
        print(' ---------- Dataset Info Start ----------')
        for name in ds_name_list:
            print(name)
        print(' ---------- Dataset Info End ----------')

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
















