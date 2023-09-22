import numpy as np

from torch.utils.data import Dataset, DataLoader

from torchvision.io import read_image

import glob
import re
from natsort import natsorted

import torch.nn as nn
import torch

import torchsummary
from matplotlib import pyplot as plt

TRAIN_SET_DIR = "./trainset/*.png"
TEST_SET_DIR = "./testset/*.png"

BATCH_SIZE = 20
EPOCH = 25
TRAIN = False


# 数据集制作类，继承自torch.utils.data.Dataset
class EXPDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


# 若为False，要确保当前目录下存在模型的pt文件和训练数据npy文件

# 卷积网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.fc = nn.Linear(128, 28)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x


class Exp:
    def __init__(self):
        self.test_loader = None
        self.train_loader = None

        self.train_size = 13440
        self.test_size = 3360

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 制作tensor数据集，生成dataloader
        self.make_dataset()

        # 实例化模型
        self.module = Net().to(self.device)
        # 交叉熵损失
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # 训练数据记录
        self.train_loss = np.zeros(EPOCH)
        self.test_loss = np.zeros_like(self.train_loss)
        self.test_acc = np.zeros_like(self.train_loss)

        # 参数量分析，生成优化器
        print("====== " + type(self.module).__name__ + " ======")
        torchsummary.summary(self.module, (1, 32, 32))
        # 随机梯度下降优化器
        self.optim = torch.optim.Adam(self.module.parameters(), lr=0.01)

        self.epoch = 0
        self.best_epoch = 0
        self.best_test_loss = float('inf')

    def make_dataset(self):
        # 读数据集
        files_train = glob.glob(TRAIN_SET_DIR)
        files_train = natsorted(files_train)

        files_test = glob.glob(TEST_SET_DIR)
        files_test = natsorted(files_test)

        train_data = torch.zeros((len(files_train), 1, 32, 32))
        train_label = torch.zeros((len(files_train)))
        test_data = torch.zeros((len(files_test), 1, 32, 32))
        test_label = torch.zeros((len(files_test)))
        for file, i in zip(files_train, range(len(files_train))):
            train_label[i] = int(re.search(r'label_(\d+)', file).group(1)) - 1
            train_data[i] = read_image(file).float() / 255

        for file, i in zip(files_test, range(len(files_test))):
            test_label[i] = int(re.search(r'label_(\d+)', file).group(1)) - 1
            test_data[i] = read_image(file).float() / 255
        train_set = EXPDataset(train_data, train_label)
        test_set = EXPDataset(test_data, test_label)

        self.train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    def plot(self):
        plt.figure(figsize=(10, 4))

        plt.subplot(121)
        plt.plot(self.train_loss, linewidth=1.0, linestyle='--', label="train")
        plt.plot(self.test_loss, linewidth=1.0, label="test")
        plt.legend(prop={'size': 12})
        plt.xlabel("epoch")
        plt.title('Loss')

        plt.subplot(122)
        plt.plot(self.test_acc, linewidth=1.0, label=" test")
        plt.legend(prop={'size': 12})
        plt.xlabel("epoch")
        plt.ylabel("%")
        plt.title('Test Accuarcy')

        plt.savefig('figure.svg', dpi=600, format='svg')
        plt.show()

    def __call__(self):
        if TRAIN:
            # 训练模型
            print("====== " + type(self.module).__name__ + " ======")
            for epoch in range(1, EPOCH + 1):
                self.epoch = epoch
                self.train()
                self.test()
            # 保存参数
            np.save('train_loss', self.train_loss)
            np.save('test_loss', self.test_loss)
            np.save('test_acc', self.test_acc)
        else:
            # 加载训练数据
            try:
                self.train_loss = np.load('train_loss.npy')
                self.test_loss = np.load('test_loss.npy')
                self.test_acc = np.load('test_acc.npy')
                self.plot()
            except FileNotFoundError:
                print("没有找到训练记录npy文件")

            try:
                self.module.load_state_dict(torch.load(type(self.module).__name__ + '.pt', map_location=self.device))
                self.test()
            except FileNotFoundError:
                print("没有找到模型参数字典pt文件")

    def train(self):
        # 在训练集上训练
        self.module.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optim.zero_grad()
            output = self.module(data)
            loss = self.criterion(output, target.long())
            train_loss += loss.item()
            loss.backward()
            self.optim.step()
        train_loss /= self.train_size
        print('Epoch: ', self.epoch)

        print('Train set: Average loss: {:.4f}'.format(train_loss))
        self.train_loss[self.epoch - 1] = train_loss

    def test(self):
        # 在测试集上测试
        self.module.eval()  # 模型进入eval模式，取消暂退
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.module(data)
                test_loss += self.criterion(output, target.long()).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= self.test_size

        if TRAIN:
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                torch.save(self.module.state_dict(), type(self.module).__name__ + ".pt")
                self.best_epoch = self.epoch
        accuracy = 100. * correct / self.test_size
        if TRAIN:
            self.test_loss[self.epoch - 1] = test_loss
            self.test_acc[self.epoch - 1] = accuracy
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, accuracy))
        print("------------")


if __name__ == "__main__":
    exp = Exp()
    exp()
