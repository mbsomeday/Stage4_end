import sys, os, argparse, shutil, time

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


class EarlyStopping_CycleGAN():
    '''
        当 loss_G 稳定不变 patience 个 epoch 时，结束训练
    '''

    def __init__(self, from_ds_name, to_ds_name, loss_G, loss_D_A, loss_D_B, save_base_dir, save_counter=None,
                 patience=10, delta=0.0001):

        self.from_ds_name = from_ds_name
        self.to_ds_name = to_ds_name
        self.loss_G = loss_G
        self.loss_D_A = loss_D_A
        self.loss_D_B = loss_D_B
        self.save_base_dir = save_base_dir
        self.patience = patience
        self.delta = delta
        self.counter = 0  # 记录loss不变的epoch数目
        self.early_stop = False
        self.save_counter = save_counter
        print('创建early stopping')

    def __call__(self, loss_G, loss_D_A, loss_D_B, netG_A2B, netG_B2A, netD_A, netD_B,
                 optimizer_G, optimizer_D_A, optimizer_D_B, epoch):
        '''
            call在每一个epoch训练结束后调用
        '''
        # 表现没有超过最好的
        if loss_G > self.loss_G + self.delta:
            self.counter += 1
            print(f'Epoch {epoch} performance:{loss_G}, best loss_G:{self.loss_G}, not save, current patience: {self.counter}')
            if self.counter > self.patience:
                self.early_stop = True

        # 表现超过最好的
        elif loss_G < self.loss_G:
            print(f'Epoch {epoch} loss_G decrease: {self.loss_G:.6f} -> {loss_G:.6f}, Save Model.')
            self.save_checkpoint(loss_G, loss_D_A, loss_D_B, netG_A2B, netG_B2A, netD_A, netD_B,
                                 optimizer_G, optimizer_D_A, optimizer_D_B, epoch)
            self.loss_G = loss_G
            self.counter = 0

        if self.save_counter:
            counter_txt = os.path.join(self.save_counter, 'counter.txt')
            with open(counter_txt, 'a') as f:
                # 获得当前时间时间戳
                now = int(time.time())
                # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
                timeArray = time.localtime(now)
                otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

                msg = str(self.counter) + ' ' + otherStyleTime + '\n'

                f.write(msg)

    def save_checkpoint(self, loss_G, loss_D_A, loss_D_B, netG_A2B, netG_B2A, netD_A, netD_B,
                        optimizer_G, optimizer_D_A, optimizer_D_B, epoch):

        # 判断是否需要删除多余的文件
        temp_total_saved_dir = os.listdir(self.save_base_dir)
        total_saved_dir = []

        # 因为云端运行的话有个 .virtual_documents，所以要做一步处理
        for cur_file in temp_total_saved_dir:
            if cur_file != '.virtual_documents':
                total_saved_dir.append(cur_file)

        # 若已经存储了 n 次了，则考虑删除
        if len(total_saved_dir) > 5:
            total_saved_dir.sort(key=lambda w: int(w.split(os.sep)[-1]), reverse=False)
            del_dir_path = total_saved_dir[0]
            print(f'del dir:{del_dir_path}')
            shutil.rmtree(del_dir_path)

        # 创建本次用于存储权重的文件夹
        save_dir_path = os.path.join(self.save_base_dir, str(epoch))
        if not os.path.exists(save_dir_path):
            os.mkdir(save_dir_path)

        # 保存 netG_A2B
        netG_A2B_save_name = f"netG_A2B-{self.from_ds_name}to{self.to_ds_name}-{epoch:02d}-{loss_G:.4f}.pth"
        netG_A2B_save_path = os.path.join(save_dir_path, netG_A2B_save_name)

        torch.save({'epoch': epoch,
                    'model_state_dict': netG_A2B.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    # 'lr': optimizer_G.state_dict()['param_groups'][0]['lr'],
                    'loss_G': loss_G,
                    'loss_D_A': loss_D_A,
                    'loss_D_B': loss_D_A
                    }
                   , netG_A2B_save_path)

        # 保存 netG_B2A
        netG_B2A_save_name = f"netG_B2A-{self.from_ds_name}to{self.to_ds_name}-{epoch:02d}-{loss_G:.4f}.pth"
        netG_B2A_save_path = os.path.join(save_dir_path, netG_B2A_save_name)
        torch.save({'epoch': epoch, 'model_state_dict': netG_B2A.state_dict()}, netG_B2A_save_path)

        # 保存 netD_A
        netD_A_save_name = f"netD_A-{self.from_ds_name}to{self.to_ds_name}-{epoch:02d}-{loss_D_A:.4f}.pth"
        netD_A_save_path = os.path.join(save_dir_path, netD_A_save_name)
        torch.save({'epoch': epoch,
                    'model_state_dict': netD_A.state_dict(),
                    'optimizer_D_A': optimizer_D_A.state_dict(),
                    }, netD_A_save_path)

        # 保存 netD_B
        netD_B_save_name = f"netD_B-{self.from_ds_name}to{self.to_ds_name}-{epoch:02d}-{loss_D_B:.4f}.pth"
        netD_B_save_path = os.path.join(save_dir_path, netD_B_save_name)
        torch.save({'epoch': epoch,
                    'model_state_dict': netD_B.state_dict(),
                    'optimizer_D_B': optimizer_D_B.state_dict(),
                    }, netD_B_save_path)

        print(f'模型在 epoch {epoch} 成功保存到{save_dir_path}')






