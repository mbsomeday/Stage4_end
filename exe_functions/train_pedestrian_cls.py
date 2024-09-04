import sys, os, argparse

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from torch.utils.data import Dataset, DataLoader


from cv_models import VARS_LOCAL, VARS_CLOUD, DEVICE
from cv_models.vgg import vgg16_bn
from training.train import train_ped_cls
from dataset.dataset import pedCls_Dataset


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name_list', nargs='+', default=['D1'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--var_opt', type=str, default='CLOUD')
    parser.add_argument('--model_name', type=str, default='VGG16')
    parser.add_argument('--model_save_dir', type=str)


    args = parser.parse_args()

    return args


opts = get_opt()

ds_name_list = opts.ds_name_list
batch_size = opts.batch_size
var_opt = opts.var_opt
model_name = opts.model_name
model_save_dir = opts.model_save_dir

# 这里保证每次只有一个列表
dataset_name = ds_name_list[0]

if var_opt == 'CLOUD':
    runOn = VARS_CLOUD
else:
    runOn = VARS_LOCAL

opt_dict = {
    'batch_size': opts.batch_size
}

model = vgg16_bn()

print(' ---------- Setting Info Start Training Pedestrian Classification ----------')
print('Datasets are: ')
for name in ds_name_list:
    print(name)
print('model_name:', model_name)
print('Batch_size:', opt_dict['batch_size'])
print('model_save_dir:', model_save_dir)
print(' ---------- Setting Info End ----------')

train_dataset = pedCls_Dataset(runOn=runOn, ds_name_list=ds_name_list, txt_name='augmentation_train.txt')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = pedCls_Dataset(runOn=runOn, ds_name_list=ds_name_list, txt_name='val.txt')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_ped_cls(model, model_name, dataset_name, train_dataset, train_loader, val_dataset, val_loader,
              model_save_dir)

















