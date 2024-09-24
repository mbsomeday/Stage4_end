import sys, os, argparse

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cv_models.cycleGAN import get_initGenerator, save_image_tensor
from cv_models import DEVICE, VARS_LOCAL, VARS_CLOUD
from dataset.dataset import pedCls_Dataset


def gen_biased_image(runOn, gen_model, org_ds_name, txt_name, batch_size, gen_image_save_dir):
    '''
        将 D1,D2,D3 转换为 D4
    :param gen_weights_path: 训练好的 image transfer 的 generator
    :param org_ds_name: 从哪一个数据集转换为D4
    :param txt_name:
    :param gen_image_save_dir: 生成图片保存的文件夹
    :return:
    '''

    # 加载 generator 模型
    generator = get_initGenerator()

    gen_weights_path = runOn['weights'][gen_model]

    checkpoints = torch.load(gen_weights_path, map_location=torch.device(DEVICE))
    generator.load_state_dict(checkpoints['model_state_dict'])

    generator.to(DEVICE)
    generator.eval()

    print(f'Using {gen_weights_path} to generate images from {org_ds_name} - {txt_name}')
    print(f'Save dir: {gen_image_save_dir}')

    # 加载 origin数据集的dataset
    org_dataset = pedCls_Dataset(runOn, ds_name_list=[org_ds_name], txt_name=txt_name)
    org_loader = DataLoader(org_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for images, labels, names in tqdm(org_loader):

            images = images.to(DEVICE)

            cur_name = names[0]

            cur_name = cur_name.replace('\\', os.sep)
            name_contents = cur_name.split(os.sep)

            # 判断在该位置是否为ped和nonped
            temp = name_contents[-2]
            img_time = ''
            if temp == 'pedestrian' or temp == 'nonPedestrian':
                obj_cls = temp
            else:
                obj_cls = name_contents[-3]
                img_time = temp

            individual_name = name_contents[-1]

            out = generator(images)

            # 如果存储路径文件夹没有创建，则创建
            save_dir_path = os.path.join(gen_image_save_dir, obj_cls, img_time)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)

            save_name = os.path.join(save_dir_path, individual_name)

            # 如果是augmentation train
            # # save_name = os.path.join(gen_image_save_dir, 'augmentation_train',obj_cls, used_name)
            #
            save_image_tensor(out, save_name)
            # break


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_model', type=str)
    parser.add_argument('--org_ds_name', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--txt_name', type=str, default='augmentation_train.txt')
    parser.add_argument('--running_on', type=str, default='CLOUD')
    parser.add_argument('--gen_image_save_dir', type=str)

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    opts = get_opt()

    var_opt = opts.running_on
    gen_model = opts.gen_model
    org_ds_name = opts.org_ds_name
    txt_name = opts.txt_name
    gen_image_save_dir = opts.gen_image_save_dir
    batch_size = opts.batch_size

    if var_opt == 'CLOUD':
        runOn = VARS_CLOUD
    else:
        runOn = VARS_LOCAL

    gen_biased_image(runOn=runOn, gen_model=gen_model,
                     org_ds_name=org_ds_name, txt_name=txt_name,
                     batch_size=batch_size,
                     gen_image_save_dir=gen_image_save_dir
                     )












