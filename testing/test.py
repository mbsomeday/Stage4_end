import sys, os, argparse

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt    # 绘图库
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from cv_models.vgg import vgg16_bn
from cv_models import VARS_LOCAL, VARS_CLOUD, DEVICE
from dataset.dataset import dsCls_Dataset, pedCls_Dataset


def datasetCls(runOn, ds_name_list, ds_label_list, txt_name, opt_dict):
    '''
        对数据集进行数据集分类
    '''

    print(' ---------- Start Dataset Classification ---------- ')

    # 模型准备
    ds_weights_path = runOn['dsCls_weights']
    model = vgg16_bn(4)
    checkpoints = torch.load(ds_weights_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    batch_size = opt_dict['batch_size']

    # 数据准备
    test_dataset = dsCls_Dataset(runOn, ds_name_list, ds_label_list, txt_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'model: {ds_weights_path}')
    print(f'batch_size: {batch_size}')
    print('Total samples:', len(test_dataset))

    # 开始检验
    correct_num = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels, name = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(images)
            _, pred = torch.max(out, 1)
            correct_num += (pred == labels).sum()
            # if pred != target_label:
            #     print(name, pred)
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    test_accuracy = correct_num / len(test_dataset)
    cm = confusion_matrix(y_true, y_pred, labels=ds_label_list)
    bc = balanced_accuracy_score(y_true, y_pred)

    print(f'Dataset Classification balanced accuracy: {bc:.4f}, accuracy: {test_accuracy:.4f}, detail:{correct_num}/{len(test_dataset)}')
    print("cm:\n", cm)


def pedestrianCls(runOn, model_weights, ds_name_list, txt_name, opt_dict):
    print(' ---------- Start Pedestrian Classification ---------- ')

    # 模型准备
    ped_weights_path = runOn['weights'][model_weights]
    model = vgg16_bn(2)
    checkpoints = torch.load(ped_weights_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    batch_size = opt_dict['batch_size']

    test_dataset = pedCls_Dataset(var_opt=runOn, ds_name_list=ds_name_list, txt_name=txt_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'model: {ped_weights_path}')
    print(f'batch_size: {batch_size}')
    print('Total samples:', len(test_dataset))

    # 开始检验
    correct_num = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels, name = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(images)
            _, pred = torch.max(out, 1)
            correct_num += (pred == labels).sum()
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    test_accuracy = correct_num / len(test_dataset)
    bc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=['nonPed', 'ped'])

    print(f'Pedestrian Classification balanced accuracy: {bc:.4f}, accuracy: {test_accuracy:.4f}, detail:{correct_num}/{len(test_dataset)}')
    print("cm:\n", cm)







