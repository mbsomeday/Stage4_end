import sys, os, argparse

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from testing.test import pedestrianCls
from cv_models import VARS_LOCAL, VARS_CLOUD, DEVICE


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name_list', nargs='+', default=['D4'])
    parser.add_argument('--txt_name', type=str, default='test.txt')
    parser.add_argument('--model_weights', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--var_opt', type=str, default='LOCAL')

    args = parser.parse_args()

    return args


opts = get_opt()

ds_name_list = opts.ds_name_list
model_weights = opts.model_weights
txt_name = opts.txt_name
var_opt = opts.var_opt

if var_opt == 'CLOUD':
    runOn = VARS_CLOUD
else:
    runOn = VARS_LOCAL

opt_dict = {
    'batch_size': opts.batch_size
}

print(' ---------- Setting Info Start ----------')
print('Datasets are: ')
for name in ds_name_list:
    print(name)
print('txt_name:', txt_name)
print('Batch_size:', opt_dict['batch_size'])
print(' ---------- Setting Info End ----------')

pedestrianCls(runOn, model_weights, ds_name_list, txt_name, opt_dict)










