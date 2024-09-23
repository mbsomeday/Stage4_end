from torch.utils.data import DataLoader
import torch, itertools, argparse, sys, os
import numpy as np

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from cv_models import VARS_LOCAL, DEVICE, VARS_CLOUD
from cv_models.cycleGAN import *
from dataset.dataset import cycleGAN_Dataset
from training.train_CycleGAN import ImagePool, train_one_epoch, val_cycleGAN
from training.strategy import EarlyStopping_CycleGAN



# ------------------------ 【函数】加载初始模型 ------------------------
def get_init_CycleGAN():
    norm_layer = get_norm_layer(norm_type='batch')
    netG_A2B = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    netG_B2A = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    netD_A = Discriminator(3)
    netD_B = Discriminator(3)

    return netG_A2B, netG_B2A, netD_A, netD_B

def init_Hyperparameter(netG_A2B, netG_B2A, netD_A, netD_B):
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    start_epoch = 0
    loss_G = np.inf
    loss_D_A = np.inf
    loss_D_B = np.inf

    return [optimizer_G, optimizer_D_A, optimizer_D_B], [loss_G, loss_D_A, loss_D_B], start_epoch,


# ------------------------ 【函数】加载训练了一半的模型 ------------------------
def reload_weights(model, weights_path):
    checkpoints = torch.load(weights_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(DEVICE)
    return model

def load_optimizer_loss(loaded_optimizer, weights_path, key_opt, key_loss, get_start_epoch=False):
    checkpoints = torch.load(weights_path, map_location=torch.device(DEVICE))
    loaded_optimizer.load_state_dict(checkpoints[key_opt])
    for state in loaded_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    loaded_loss = checkpoints[key_loss]
    start_epoch = checkpoints['epoch']
    if get_start_epoch:
        return loaded_optimizer, loaded_loss, start_epoch
    else:
        return loaded_optimizer, loaded_loss

def reload(preTrainedWeights):
    # 先确定各个模型的权重
    netG_A2B_weights = preTrainedWeights[0]
    netG_B2A_weights = preTrainedWeights[1]
    netD_A_weights = preTrainedWeights[2]
    netD_B_weights = preTrainedWeights[3]

    # load model
    norm_layer = get_norm_layer(norm_type='batch')
    netG_A2B = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    netG_B2A = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)

    netG_A2B = reload_weights(netG_A2B, netG_A2B_weights)
    netG_B2A = reload_weights(netG_B2A, netG_B2A_weights)

    netD_A = Discriminator(3)
    netD_B = Discriminator(3)
    netD_A = reload_weights(netD_A, netD_A_weights)
    netD_B = reload_weights(netD_B, netD_B_weights)

    # load optimizer
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    optimizer_G, loss_G, start_epoch = load_optimizer_loss(optimizer_G, netG_A2B_weights,
                                                      key_opt='optimizer_G', key_loss='loss_G',
                                                      get_start_epoch=True)
    optimizer_D_A, loss_D_A = load_optimizer_loss(optimizer_D_A, netD_A, key_opt='optimizer_D_A', key_loss='loss_D_A', get_start_epoch=False)
    optimizer_D_B, loss_D_B = load_optimizer_loss(optimizer_D_B, netD_B, key_opt='optimizer_D_B', key_loss='loss_D_B', get_start_epoch=False)

    optimizers = [optimizer_G, optimizer_D_A, optimizer_D_B]
    losses = [loss_G, loss_D_A, loss_D_B]

    model = [netG_A2B, netG_B2A, netD_A, netD_B]

    return start_epoch, model, optimizers, losses


# 以下为train部分的code

# ------------------------ argparse 传参 ------------------------

pretrained = False
pool_size = 50
EPOCHS = 100

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--ds_name_list', nargs='+', default=['D4', 'D4'])
    parser.add_argument('--save_base_dir', type=str, default=r'/kaggle/working/model')
    parser.add_argument('--var_opt', type=str, default='LOCAL')
    parser.add_argument('--preTrainedWeights', nargs='+', default=[])
    parser.add_argument('--get_num_train', default=10000)


    args = parser.parse_args()

    return args

# ------------------------ 加载模型 ------------------------

def train(preTrainedWeights, get_num_train):

    if len(preTrainedWeights) == 0:
        models = get_init_CycleGAN()
        netG_A2B, netG_B2A, netD_A, netD_B = models
        optimizers, losses, start_epoch = init_Hyperparameter(netG_A2B, netG_B2A, netD_A, netD_B)

    # 加载训练了一半的模型
    else:
        start_epoch, models, optimizers, losses = reload(preTrainedWeights)
        netG_A2B, netG_B2A, netD_A, netD_B = models

    netG_A2B.to(DEVICE)
    netG_B2A.to(DEVICE)
    netD_A.to(DEVICE)
    netD_B.to(DEVICE)

    optimizer_G, optimizer_D_A, optimizer_D_B = optimizers
    loss_G, loss_D_A, loss_D_B = losses


    # ------------------------ 加载数据 ------------------------
    train_dataset = cycleGAN_Dataset(runOn, dataset_name_list=ds_name_list, txt_name='augmentation_train.txt', get_num=get_num_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = cycleGAN_Dataset(runOn, dataset_name_list=ds_name_list, txt_name='val.txt')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    # ------------------------ 加载其他训练相关 ------------------------
    fake_A_pool = ImagePool(pool_size)
    fake_B_pool = ImagePool(pool_size)

    criterionGAN = torch.nn.MSELoss()
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()

    early_stopping = EarlyStopping_CycleGAN(from_ds_name=ds_name_list[0],
                                            to_ds_name=ds_name_list[1], save_base_dir=save_base_dir,
                                            loss_G=loss_G, loss_D_A=loss_D_A, loss_D_B=loss_D_B
                                            )



    # ------------------------ 输出一些基本信息 ------------------------
    print('-' * 20 + 'Training Info' + '-' * 20)
    print('Total Training Samples:', len(train_dataset))
    print('Total Batch:', len(train_loader))
    print(f'Start Epoch:{start_epoch}, Total EPOCH:', EPOCHS)


    for epoch in range(start_epoch, EPOCHS):
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B = train_one_epoch(runOn,
                                                                                                        netG_A2B, netG_B2A,
                                                                                                        netD_A, netD_B,
                                                                                                        fake_A_pool,
                                                                                                        fake_B_pool,
                                                                                                        criterionGAN,
                                                                                                        criterionCycle,
                                                                                                        criterionIdt,
                                                                                                        optimizer_G,
                                                                                                        optimizer_D_A,
                                                                                                        optimizer_D_B,
                                                                                                        epoch,
                                                                                                        train_dataset,
                                                                                                        train_loader)

        loss_G, loss_D_A, loss_D_B = val_cycleGAN(netG_A2B, netG_B2A, netD_A, netD_B,
                                                  val_dataset, val_loader,
                                                  criterionGAN, criterionCycle, criterionIdt,
                                                  )

        # Early Stopping 策略
        early_stopping(loss_G, loss_D_A, loss_D_B, netG_A2B, netG_B2A, netD_A, netD_B,
                       optimizer_G, optimizer_D_A, optimizer_D_B, epoch=epoch + 1)

        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

        # 每次重新加载dataloader
        train_dataset = cycleGAN_Dataset(runOn, dataset_name_list=ds_name_list, txt_name='augmentation_train.txt', get_num=get_num_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)




if __name__ == '__main__':
    opts = get_opt()

    ds_name_list = opts.ds_name_list
    save_base_dir = opts.save_base_dir
    var_opt = opts.var_opt
    batch_size = opts.batch_size
    preTrainedWeights = opts.preTrainedWeights
    get_num_train = opts.get_num_train

    if var_opt == 'CLOUD':
        runOn = VARS_CLOUD
    else:
        runOn = VARS_LOCAL

    train(preTrainedWeights, get_num_train)




