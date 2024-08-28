import functools, torch, os, math, random
from tqdm import tqdm
import numpy as np
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
from torch.utils.data import Dataset, DataLoader
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
import torch.nn.functional as F

from cv_models.ResNet import resnet34
from strategy import EarlyStopping
from cv_models import DEVICE, VARS_LOCAL
from dataset.dataset import pedCls_Dataset


def train_one_epoch(model, loss_fn, optimizer, epoch, train_dataset, train_loader):
    model.train()

    training_loss = 0.0
    training_correct_num = 0
    start_time = time()

    # 用于 balanced accuracy
    y_pred = []
    y_true = []

    for batch, data in enumerate(tqdm(train_loader)):
        # 将image和label放到GPU中
        images, labels, _ = data
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        out = model(images)
        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        _, pred = torch.max(out, 1)
        training_correct_num += (pred == labels).sum()

    # 用 balanced accuracy来检验
    training_accuracy = balanced_accuracy_score(y_true, y_pred)
    unit_training_loss = training_loss / len(train_dataset)

    # training_accuracy = training_correct_num / len(train_dataset)

    print(
        f'Training time for epoch:{epoch + 1}: {(time() - start_time):.2f}s, total training loss:{training_loss:.4f}, unit training loss:{unit_training_loss:.8f},\n training accuracy:{training_accuracy:.4f}')
    return model


def val_model(model, loss_fn, val_dataset, val_loader):
    model.eval()
    val_loss = 0.0
    val_correct_num = 0

    # 用于 balanced accuracy
    y_pred = []
    y_true = []

    with torch.no_grad():
        for data in tqdm(val_loader):
            images, labels, _ = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(images)
            loss = loss_fn(out, labels)
            _, pred = torch.max(out, 1)
            val_correct_num += (pred == labels).sum()

            # 用于 balanced accuracy
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            val_loss += loss.item()

    # 用 balanced accuracy来检验
    bc = balanced_accuracy_score(y_true, y_pred)
    unit_val_loss = val_loss / len(val_dataset)

    # val_accuracy = val_correct_num / len(val_dataset)
    print('Val Loss:{:.4f}, unit val loss:{:.8f}, balanced accuracy:{:.4f}'.format(val_loss, unit_val_loss, bc))

    return val_loss, bc


def train_ped_cls(model, model_name, dataset_name, train_dataset, train_loader, val_dataset, val_loader,
          model_save_dir):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.initialize_weights()
    start_epoch = 0

    loss_fn = torch.nn.CrossEntropyLoss()

    model = model.to(DEVICE)
    EPOCHS = 100

    print('-' * 20 + 'Training Info' + '-' * 20)
    print('Total Training Samples:', len(train_dataset))
    print('Total Batch:', len(train_loader))
    print('Total EPOCH:', EPOCHS)

    print('-' * 20 + 'Validation Info' + '-' * 20)
    print('Total Val Samples:', len(val_dataset))

    formatted_datasetname = ''
    for idx, cur_ds_name in enumerate(dataset_name):
        if idx == 0:
            formatted_datasetname += cur_ds_name
        else:
            formatted_datasetname += 'and' + cur_ds_name

    early_stopping = EarlyStopping(model_name=model_name, dataset_name=formatted_datasetname,
                                   model_save_dir=model_save_dir,
                                   )

    for epoch in range(start_epoch, EPOCHS):
        print('-' * 30 + 'begin EPOCH ' + str(epoch + 1) + '-' * 30)
        model = train_one_epoch(model, loss_fn, optimizer, epoch, train_dataset, train_loader)
        val_loss, val_accuracy = val_model(model, loss_fn, val_dataset, val_loader)

        # Early Stopping 策略
        early_stopping(val_loss=val_loss, val_acc=val_accuracy, model=model, optimizer=optimizer, epoch=epoch + 1)
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

        print('*' * 50)



if __name__ == '__main__':
    model = resnet34(pretrained=False)
    model_name = 'resNet34'
    dataset_name = 'D1'
    runOn = VARS_LOCAL
    train_dataset = pedCls_Dataset(runOn=runOn, ds_name_list=['D1'], txt_name='train.txt')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    val_dataset = pedCls_Dataset(runOn=runOn, ds_name_list=['D1'], txt_name='val.txt')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model_save_dir = r'D:\my_phd\on_git\Stage4_end\cache'

    train_ped_cls(model, model_name, dataset_name, train_dataset, train_loader, val_dataset, val_loader,
          model_save_dir)















