import torch
import torch.nn as nn
import torch, argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=r'/kaggle/working/', type=str)

    args = parser.parse_args()

    return args

opts = get_opt()
data_dir = opts.data_dir


# PyTorch中的所有神经网络模型都应该继承自nn.Module基类，并进行初始化。
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # 编码器部分，高维->低维。
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),  # 第一层全连接层，将输入的784维数据（即28*28像素的图像展平成向量）压缩到128维。
            nn.ReLU(),  # 激活函数ReLU，用于增加网络的非线性，帮助模型学习复杂的特征。
            nn.Linear(128, 64),  # 第二层全连接层，进一步将数据从128维压缩到64维。
            nn.ReLU(),  # 再次使用ReLU激活函数。
            nn.Linear(64, 12),  # 第三层全连接层，将数据从64维压缩到12维。
            nn.ReLU(),  # 再次使用ReLU激活函数。
            nn.Linear(12, 3)  # 最后一层全连接层，将数据最终压缩到3维，得到编码后的数据。
        )

        # 解码器部分，低维->高维。
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),  # 第一层全连接层，将编码后的3维数据扩展到12维。
            nn.ReLU(),  # 使用ReLU激活函数。
            nn.Linear(12, 64),  # 第二层全连接层，将数据从12维扩展到64维。
            nn.ReLU(),  # 再次使用ReLU激活函数。
            nn.Linear(64, 128),  # 第三层全连接层，将数据从64维扩展到128维。
            nn.ReLU(),  # 再次使用ReLU激活函数。
            nn.Linear(128, 28 * 28),  # 最后一层全连接层，将数据从128维扩展回784维，即原始图像大小。
            nn.Sigmoid()  # 使用Sigmoid激活函数，将输出压缩到0到1之间，因为原始图像在输入之前经过了标准化。
        )

    def forward(self, x):
        x = self.encoder(x)  # 将输入数据通过编码器压缩。
        x = self.decoder(x)  # 然后通过解码器进行重构。
        return x  # 返回重构的数据。


# 如果有GPU可用，优先使用GPU, 否则使用CPU运算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# 定义图像预处理步骤
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为PyTorch张量,并将每个像素值从[0, 255]范围内缩放到[0.0, 1.0]
])

# 自动下载并加载MNIST数据集，并应用定义好的预处理步骤
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

# 用于批量加载数据，每批数据包含32个样本，并在每个epoch随机打乱数据
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = Autoencoder()  # 实例化自编码器（AE）模型
model.to(device)  # 如果cuda可用，则将模型从CPU移动到GPU上进行计算

criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器，学习率为0.001

num_epochs = 20  # 训练周期
lowest_loss = float('inf')  # 初始化最低损失为正无穷，用于跟踪保存最好的模型




for epoch in range(num_epochs):
    total_loss = 0.0
    for data in train_loader:
        # 自编码器只要图像数据，不需要标签.
        # 图像张量形状：[batch_size, 1, 28, 28]
        # 其中，batch_size由train_loader指定，28为Mnist数据集图像的宽和高，1代表图像只有一个通道，是黑白图片。（彩色图片有RGB三个通道）
        img, _ = data
        img = img.to(device)  # 将数据从CPU移动到指定的设备

        img = img.view(img.size(0), -1)  # 将图像数据展平 形状：[batch_size, 28 * 28]
        output = model(img)  # 通过模型前向传播得到重建的输出 形状：[batch_size, 28 * 28]
        loss = criterion(output, img)  # 计算重建图像与原图之间的损失

        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新模型参数

        total_loss += loss.item()  # 累加损失

    avg_loss = total_loss / len(train_loader)  # 计算这个epoch的平均损失

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # 如果损失是目前为止最低的，保存模型
    if avg_loss < lowest_loss:
        lowest_loss = avg_loss
        torch.save(model.state_dict(), 'best.pth')  # 保存模型
        print(f'New lowest average loss {lowest_loss:.4f} at epoch {epoch + 1}, model saved.')






















