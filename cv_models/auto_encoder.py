import sys, os

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import torch, torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cv_models import DEVICE


train_data = torchvision.datasets.MNIST(root=r'/kaggle/working',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True
                                        )

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

class autoEncoder(nn.Module):
    def __init__(self):
        super(autoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

model = autoEncoder()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(100):
    for data in train_loader:
        img, _ = data
        img = img.to(DEVICE)
        img = img.view(img.size(0), -1)

        # forward
        output = model(img)
        loss = loss_func(output, img)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 100, loss.item()))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, r'/kaggle/working/mlp_img/image_{}.png'.format(epoch))

        pic1 = to_img(img.data)
        save_image(pic, r'/kaggle/working/org_img/Ori_image_{}.png'.format(epoch))























