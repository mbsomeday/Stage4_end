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

from cv_models.vgg import vgg16_bn
from cv_models import VARS_LOCAL, VARS_CLOUD, DEVICE
from dataset.dataset import dsCls_Dataset


def datasetCls_all(var_opt, ds_name_list, ds_label_list, txt_name, opt_dict):
    '''
        对全部数据进行数据集分类
    '''

    # 模型准备
    ds_weights_path = var_opt['dsCls_weights']
    model = vgg16_bn(4)
    checkpoints = torch.load(ds_weights_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    batch_size = opt_dict['batch_size']

    # 数据准备
    test_dataset = dsCls_Dataset(var_opt, ds_name_list, ds_label_list, txt_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    cm = confusion_matrix(y_true, y_pred)

    print("cm:\n", cm)
    print(f'Dataset Classification accuracy on all dataset: {test_accuracy:.4f}, detail:{correct_num}/{len(test_dataset)}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name_list', nargs='+', default=['D1', 'D2', 'D3', 'D4'])
    parser.add_argument('--ds_label_list', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--txt_name', type=str, default='test.txt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--var_opt', type=str, default='CLOUD')

    args = parser.parse_args()

    ds_name_list = args.ds_name_list
    ds_label_list = args.ds_label_list
    txt_name = args.txt_name
    var_opt = args.var_opt

    if var_opt == 'CLOUD':
        var_opt = VARS_CLOUD
    else:
        var_opt = VARS_LOCAL

    opt_dict = {
        'batch_size': args.batch_size
    }

    datasetCls_all(var_opt, ds_name_list, ds_label_list, txt_name, opt_dict)










