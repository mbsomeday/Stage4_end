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
        'D1': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D1-014-0.9740.pth',
        'D2': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D2-025-0.9124.pth',
        'D3': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D3-025-0.9303.pth',
        'D4': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D4-013-0.9502.pth',

        'D1toD4': r'D:\my_phd\Model_Weights\Stage4\Generated\vgg16bn-D1toD4-006-0.9589.pth',
        'D2toD4': r'D:\my_phd\Model_Weights\Stage4\Generated\vgg16bn-D2toD4-025-0.9304.pth',
        'D3toD4': r'D:\my_phd\Model_Weights\Stage4\Generated\vgg16bn-D3toD4-047-0.9378.pth',

        'G1to4': r'D:\my_phd\Model_Weights\Stage4\CycleGAN\netG_A-D1toD4-037-0.8150.pth',
        'G2to4': r'D:\my_phd\Model_Weights\Stage4\CycleGAN\netG_A-D2toD4-044-0.7050.pth',
        'G3to4': r'D:\my_phd\Model_Weights\Stage4\CycleGAN\netG_A-D3toD4-039-0.5250.pth',
    }

}


VARS_CLOUD = {
    'D1': r'/kaggle/input/stage4-d1-ecpdaytime-7augs',
    'D2': r'/kaggle/input/stage4-d2-citypersons-7augs',
    'D3': r'/kaggle/input/stage4-d3-ecpnight-7augs',
    'D4': r'/kaggle/input/stage4-d4-7augs',

    'D1toD4': r'/kaggle/input/stage4-d1tod4-stable',
    'D2toD4': r'/kaggle/input/stage4-d2tod4-dataset-stable',
    'D3toD4': r'/kaggle/input/stage4-d3tod4-dataset-stable',

    'test': r'/kaggle/working/test',

    'dsCls_weights': r'/kaggle/input/stage4-dscls-weights/vgg16bn-dsCls-029-0.9777.pth',

    'weights': {
        'D1': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D1-014-0.9740.pth',
        'D2': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D2-025-0.9124.pth',
        'D3': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D3-025-0.9303.pth',
        'D4': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D4-013-0.9502.pth',

        'D1toD4': r'/kaggle/input/stage4-trainongentod4-weights-stable/vgg16bn-D1toD4-006-0.9589.pth',
        'D2toD4': r'/kaggle/input/stage4-trainongentod4-weights-stable/vgg16bn-D2toD4-025-0.9304.pth',
        'D3toD4': r'/kaggle/input/stage4-trainongentod4-weights-stable/vgg16bn-D3toD4-047-0.9378.pth',

        'G1to4': r'/kaggle/input/stage4-tod4generator-weights/netG_A-D1toD4-037-0.8150.pth',
        'G2to4': r'/kaggle/input/stage4-tod4generator-weights/netG_A-D2toD4-044-0.7050.pth',
        'G3to4': r'/kaggle/input/stage4-tod4generator-weights/netG_A-D3toD4-039-0.5250.pth',

        'Res34D1': r'/kaggle/input/stage4-resnet34-baseweights/resNet34-D1-015-0.9437.pth',
        'Res34D3': r'/kaggle/input/stage4-resnet34-baseweights/resNet34-D3-014-0.8933.pth'

    }
}












