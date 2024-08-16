import torch
import os


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VARS_LOCAL = {
    'D1': r'D:\my_phd\dataset\Stage3\D1_ECPDaytime',
    'D2': r'D:\my_phd\dataset\Stage3\D2_CityPersons',
    'D3': r'D:\my_phd\dataset\Stage3\D3_ECPNight',
    'D4': r'D:\my_phd\dataset\Stage3\D4_BDD100K',
    'dsCls_weights': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-dsCls-029-0.9777.pth',

    'weights': {
        'D1toD4': r'D:\my_phd\Model_Weights\Stage4\Generated',
    }

}


VARS_CLOUD = {
    'D1': r'/kaggle/input/stage4-d1-ecpdaytime-7augs',
    'D2': r'/kaggle/input/stage4-d2-citypersons-7augs',
    'D3': r'/kaggle/input/stage4-d3-ecpnight-7augs',
    'D4': r'/kaggle/input/stage4-d4-7augs',
    'dsCls_weights': r'/kaggle/input/stage4-dscls-weights/vgg16bn-dsCls-029-0.9777.pth',

    'weights': {
        'D1toD4': r'/kaggle/input/stage4_trainongentod4_weights_stable/vgg16bn-D1toD4-006-0.9589.pth'
    }
}












