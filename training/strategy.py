import sys, os, argparse

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import numpy as np
import os
import torch


class EarlyStopping():
    '''
        保存当前为止最好的模型(loss最低)，
        当loss稳定不变patience个epoch时，结束训练
    '''

    def __init__(self, model_name, dataset_name, model_save_dir, patience=15, verbose=True, delta=0.0001):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        # 存储权重的文件夹
        self.model_save_dir = model_save_dir

        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # 记录loss不变的epoch数目
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.best_score = None
        self.delta = delta
        print('创建early stopping')


    def __call__(self, val_loss, val_acc, model, optimizer, epoch):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer, epoch)

        # 表现没有超过best
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        # 比best表现好
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer, epoch)
            self.counter = 0

    # 删除多余的权重文件
    def del_redundant_weights(self):
        # 删除已经有的文件,只保留n+1个模型
        num_saved = 3
        all_weights_temp = os.listdir(self.model_save_dir)
        all_weights = []
        for weights in all_weights_temp:
            if weights.endswith('.pth'):
                all_weights.append(weights)

        print('当前已保存:', len(all_weights))

        # 按存储格式来： save_name = f"netD_A-D1toD4-{epoch + 1:03d}-{min_loss_G:.6f}.pth"
        if len(all_weights) > num_saved:
            sorted = []
            for weight in all_weights:
                lossVal = weight.split('-')[-1]
                sorted.append((weight, lossVal))

            # 如果按loss，则reverse为False
            #             sorted.sort(key=lambda w: w[1], reverse=False)
            # 如果按accuracy，则reverse为True
            sorted.sort(key=lambda w: w[1], reverse=True)

            del_path = os.path.join(self.model_save_dir, sorted[-1][0])
            os.remove(del_path)
            print('del file:', del_path)

    def save_checkpoint(self, val_acc, model, optimizer, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_max:.4f} --> {val_acc:.4f}).  Saving model ...')

        self.del_redundant_weights()
        save_name = f"{self.model_name}-{self.dataset_name}-{epoch:03d}-{val_acc:.4f}.pth"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        save_path = os.path.join(self.model_save_dir, save_name)

        # 存储权重
        torch.save(checkpoint, save_path)
        self.val_acc_max = val_acc








