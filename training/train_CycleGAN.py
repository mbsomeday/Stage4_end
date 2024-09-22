import os, torch, random, sys, itertools
from torch import nn

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from time import time
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader

from cv_models import DEVICE
from dataset.dataset import cycleGAN_Dataset
from cv_models.cycleGAN import get_norm_layer, ResnetGenerator, Discriminator
from training.strategy import EarlyStopping_CycleGAN


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


def backward_D_basic(netD, real, fake, criterionGAN, requireGrad=True):
    """Calculate GAN loss for the discriminator

    Parameters:
        netD (network)      -- the discriminator D
        real (tensor array) -- real images
        fake (tensor array) -- images generated by a generator

    Return the discriminator loss.
    We also call loss_D.backward() to calculate the gradients.
    """
    # Real
    pred_real = netD(real)
    loss_D_real = criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    if requireGrad:
        loss_D.backward()
    return loss_D


def makeWeightDir(model_name):
    '''
        创建对应的权重存储文件夹
    '''
    base = r'/kaggle/working/'
    dir_path = os.path.join(base, model_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def get_init_CycleGAN(runOn):
    '''
        获取初始化的 CycleGAN
    '''

    norm_layer = get_norm_layer(norm_type='batch')

    # 生成器
    netG_A2B = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)
    netG_B2A = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)

    # 判别器
    #     netD_A = NLayerDiscriminator(3, 64, n_layers=3, norm_layer=norm_layer)
    #     netD_B = NLayerDiscriminator(3, 64, n_layers=3, norm_layer=norm_layer)
    netD_A = Discriminator(3)
    netD_B = Discriminator(3)

    netG_A2B.to(runOn)
    netG_B2A.to(runOn)
    netD_A.to(runOn)
    netD_B.to(runOn)

    return netG_A2B, netG_B2A, netD_A, netD_B


def load_preTrained_cycleGAN(runOn, name, weights):
    '''
        name: gen / dis
        获取训练了一半的模型
    '''
    norm_layer = get_norm_layer(norm_type='batch')

    if name == 'gen':
        model = ResnetGenerator(3, 3, ngf=64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)

    else:
        model = Discriminator(3)

    checkpoints = torch.load(weights, map_location=torch.device(runOn))
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(runOn)

    return model


def get_dataset_loader(runOn, dataset_name_list, batch_size, txt_name, get_num):
    '''
        获取 dataset 和 dataloader
    '''
    get_dataset = cycleGAN_Dataset(runOn, dataset_name_list, txt_name, get_num)
    get_loader = DataLoader(get_dataset, batch_size, shuffle=True, drop_last=False)

    return get_dataset, get_loader

def train_one_epoch(runOn, netG_A2B, netG_B2A, netD_A, netD_B,
                    fake_A_pool, fake_B_pool,
                    criterionGAN, criterionCycle, criterionIdt,
                    optimizer_G, optimizer_D_A, optimizer_D_B,
                    epoch, train_dataset, train_loader):
    print('-' * 30, f'Epoch {epoch + 1}', '-' * 30)

    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

    Tensor = torch.cuda.FloatTensor if runOn == 'cuda' else torch.Tensor

    epoch_start_time = time()  # timer for entire epoch

    for batch_id, data in enumerate(tqdm(train_loader)):
        real_A = data[0].to(runOn)
        real_B = data[1].to(runOn)

        num_sample = real_A.shape[0]

        target_real = Variable(Tensor(num_sample, 1).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(num_sample, 1).fill_(0.0), requires_grad=False)

        #         target_real = torch.tensor(torch.zeros(num_sample, 1), device='cuda')
        #         target_real = torch.tensor(torch.ones(num_sample, 1), device='cuda')

        ###### Generators A2B and B2A ######

        # 先 train G_A and G_B, 此时将 discriminator 的 gradient 设置为 0
        set_requires_grad([netD_A, netD_B], False)
        optimizer_G.zero_grad()

        # Identity loss
        same_B = netG_A2B(real_B)
        loss_identity_B = criterionIdt(same_B, real_B) * 5.0
        same_A = netG_B2A(real_A)
        loss_identity_A = criterionIdt(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterionGAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterionGAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterionCycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterionCycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######

        set_requires_grad([netD_A, netD_B], True)
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterionGAN(pred_real, target_real)

        # Fake loss
        #         fake_A = fake_A_buffer.push_and_pop(fake_A)
        fake_A = fake_A_pool.query(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterionGAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterionGAN(pred_real, target_real)

        # Fake loss
        #         fake_B = fake_B_buffer.push_and_pop(fake_B)
        fake_B = fake_B_pool.query(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterionGAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

    print(f'Time for training epoch {epoch + 1}: {int(time() - epoch_start_time)} s. loss_G:{loss_G:.6f}, loss_D_A:{loss_D_A:.6f}, loss_D_B:{loss_D_B:.6f}.')

    return netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B


def val_cycleGAN(netG_A2B, netG_B2A, netD_A, netD_B, val_dataset, val_loader, criterionGAN, criterionCycle,
                 criterionIdt):
    netG_A2B.eval()
    netG_B2A.eval()
    netD_A.eval()
    netD_B.eval()

    # Buffer
    fake_A_pool = ImagePool(pool_size=50)
    fake_B_pool = ImagePool(pool_size=50)

    Tensor = torch.cuda.FloatTensor if DEVICE == 'cuda' else torch.Tensor

    with torch.no_grad():
        for batch_id, data in enumerate(tqdm(val_loader)):
            real_A = data[0].to(DEVICE)
            real_B = data[1].to(DEVICE)

            num_sample = real_A.shape[0]

            target_real = Variable(Tensor(num_sample, 1).fill_(1.0), requires_grad=False)
            target_fake = Variable(Tensor(num_sample, 1).fill_(0.0), requires_grad=False)

            #             target_real = torch.tensor(torch.zeros(num_sample, 1), device='cuda')
            #             target_real = torch.tensor(torch.ones(num_sample, 1), device='cuda')

            ###### Generators A2B and B2A ######

            # Identity loss
            same_B = netG_A2B(real_B)
            loss_identity_B = criterionIdt(same_B, real_B) * 5.0
            same_A = netG_B2A(real_A)
            loss_identity_A = criterionIdt(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterionGAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterionGAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterionCycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterionCycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            ###################################

            ###### Discriminator A ######
            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterionGAN(pred_real, target_real)

            # Fake loss
            # fake_A = fake_A_buffer.push_and_pop(fake_A)
            fake_A = fake_A_pool.query(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterionGAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            ###################################

            ###### Discriminator B ######
            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterionGAN(pred_real, target_real)

            # Fake loss
            # fake_B = fake_B_buffer.push_and_pop(fake_B)
            fake_B = fake_B_pool.query(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterionGAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            ###################################

    return loss_G, loss_D_A, loss_D_B

def train(runOn, dataset_name_list, batch_size, get_num, EPOCHS):
    # 获取 对应数据集的 name 和 label
    from_ds_name = dataset_name_list[0]
    from_ds_label = int(from_ds_name[1]) - 1

    to_ds_name = dataset_name_list[1]
    to_ds_label = int(to_ds_name[1]) - 1

    lr = 0.0002
    beta1 = 0.5

    # 初次训练
    netG_A2B, netG_B2A, netD_A, netD_B = get_init_CycleGAN(runOn)
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=lr, betas=(beta1, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    start_epoch = 0
    loss_G = np.inf
    loss_D_A = np.inf
    loss_D_B = np.inf

    # 获取 dataset
    train_dataset, train_loader = get_dataset_loader(runOn, dataset_name_list, batch_size,
                                                     txt_name='augmentation_train.txt',
                                                     get_num=get_num)
    # TODO： 这里的get_num每次要调整
    val_dataset, val_loader = get_dataset_loader(runOn, dataset_name_list, batch_size,
                                                 txt_name='val.txt', get_num=1000)

    # Buffer
    pool_size = 50
    fake_A_pool = ImagePool(pool_size)
    fake_B_pool = ImagePool(pool_size)

    # loss and optimizer
    #     criterionGAN = GANLoss('lsgan').to(DEVICE)
    criterionGAN = torch.nn.MSELoss()
    criterionCycle = torch.nn.L1Loss()
    criterionIdt = torch.nn.L1Loss()

    # 输出一些基本信息
    print('-' * 20 + 'Training Info' + '-' * 20)
    print(f'Transferring {from_ds_name} to {to_ds_name}')
    print('Total Training Samples:', len(train_dataset))
    print('Total Batch:', len(train_loader))
    print(f'Start Epoch:{start_epoch}, Total EPOCH:', EPOCHS)

    model_save_dir = r'/kaggle/working'
    early_stopping = EarlyStopping_CycleGAN(from_ds_name=from_ds_name,
                                   to_ds_name=to_ds_name,
                                   save_base_dir=model_save_dir,
                                   patience=10)

    for epoch in range(start_epoch, EPOCHS):
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()

        netG_A2B, netG_B2A, netD_A, netD_B, optimizer_G, optimizer_D_A, optimizer_D_B = train_one_epoch(runOn,
                                                                                                        netG_A2B,
                                                                                                        netG_B2A,
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
        #
        # loss_G, loss_D_A, loss_D_B = val_cycleGAN(netG_A2B, netG_B2A, netD_A, netD_B,
        #                                           val_dataset, val_loader,
        #                                           criterionGAN, criterionCycle, criterionIdt,
        #                                           )

        # Early Stopping 策略
        early_stopping(loss_G, loss_D_A, loss_D_B, netG_A2B, netG_B2A, netD_A, netD_B,
                       optimizer_G, optimizer_D_A, optimizer_D_B, epoch=epoch + 1)

        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

        # 每次重新加载dataloader
        train_dataset, train_loader = get_dataset_loader(runOn,
                                                        dataset_name_list, batch_size,
                                                         txt_name='augmentation_train.txt',
                                                         get_num=get_num)



















