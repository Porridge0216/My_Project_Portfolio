import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn import functional as F
import torchsummary
from matplotlib import pyplot as plt

BATCH_SIZE = 128
EPOCH = 50
TRAIN = False


# 若为False，要确保当前目录下存在模型的pt文件和训练数据npy文件

# 普通全连接网络
class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(28 * 28, 77)
        self.dense2 = nn.Linear(77, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        return x


# 深一层的全连接网络
class Dense2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(28 * 28, 70)
        self.dense2 = nn.Linear(70, 50)
        self.dense3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = self.dense3(x)
        return x


# 带暂退的全连接网络
class Dense2Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(28 * 28, 70)
        self.dropout1 = nn.Dropout(0.25)
        self.dense2 = nn.Linear(70, 50)
        self.dropout2 = nn.Dropout(0.25)
        self.dense3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.dense1(x))
        x = self.dropout1(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout2(x)
        x = self.dense3(x)
        return x


# 卷积网络
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)




        self.dense1 = nn.Linear(16 * 5 * 5, 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)




        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = self.dense3(x)
        return x


# 用1x1卷积代替全连接的LeNet
class NiNLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)

        # NiN
        self.nin = self.nin_block(16, 10, 3, 1, 1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def nin_block(in_channels, out_channels, kernel_size, strides, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.nin(x)
        x = self.pool3(x)

        x = x.view(-1, 10)
        return x


# Inception块
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)  # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)  # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# 带Inception块的LeNet
class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.inception = Inception(6, 4, (6, 4), (6, 4), 4)
        self.conv2 = nn.Conv2d(16, 16, 5)

        self.pool2 = nn.MaxPool2d(2)
        self.dense1 = nn.Linear(16 * 5 * 5, 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.inception(x)
        x = self.conv2(x)

        x = self.pool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = self.dense3(x)
        return x


# 残差块
class Residual(nn.Module):  # @save
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=1, stride=strides)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X

        return F.relu(Y)


# 残差网络
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.b1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                                nn.BatchNorm2d(6), nn.ReLU(),
                                nn.MaxPool2d(2))
        self.b2 = nn.Sequential(*self.resnet_block(6, 6, 2, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(6, 16, 2))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense = nn.Linear(16, 10)

    @staticmethod
    def resnet_block(input_channels, num_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x)

        x = x.view(-1, 16)
        x = self.dense(x)
        return x


class MNISTexp:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 读数据集
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = torchvision.datasets.MNIST(root=".", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.MNIST(root=".", train=False, transform=trans, download=True)

        # mnist_train = torchvision.datasets.FashionMNIST(root=".", train=True, transform=trans, download=True)
        # mnist_test = torchvision.datasets.FashionMNIST(root=".", train=False, transform=trans, download=True)

        self.train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True)

        # 加载评估的模型
        self.modules = [Dense().to(self.device),
                        Dense2().to(self.device),
                        Dense2Dropout().to(self.device),
                        LeNet().to(self.device),
                        NiNLeNet().to(self.device),
                        GoogLeNet().to(self.device),
                        ResNet().to(self.device)]

        self.optim = []
        # 交叉熵损失
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # 训练数据
        self.train_loss = np.zeros((len(self.modules), EPOCH))
        self.test_loss = np.zeros_like(self.train_loss)
        self.test_acc = np.zeros_like(self.train_loss)
        # 参数量分析，生成优化器
        for model in self.modules:
            print("====== " + type(model).__name__ + " ======")
            torchsummary.summary(model, (1, 28, 28))
            self.optim.append(optim.SGD(model.parameters(), lr=0.01, momentum=0.5))

        self.color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']

        self.epoch = 0

    def plot(self):
        plt.figure(figsize=(18, 18))
        plt.subplot(221)
        for i, model, clr in zip(range(len(self.modules)), self.modules, self.color):
            label = type(model).__name__
            plt.plot(self.train_loss[i], linewidth=1.0, color=clr, linestyle='--', label=label + " train")
            plt.plot(self.test_loss[i], linewidth=1.0, color=clr, label=label + " test")
            plt.legend(prop={'size': 12})
            plt.xlabel("epoch")
            plt.title('Loss')

        plt.subplot(222)
        for i, model, clr in zip(range(len(self.modules)), self.modules, self.color):
            label = type(model).__name__
            plt.plot(self.test_acc[i], linewidth=1.0, color=clr, label=label + " test")
            plt.legend(prop={'size': 12})
            plt.xlabel("epoch")
            plt.ylabel("%")
            plt.title('Test Accuarcy')

        plt.subplot(223)
        for i, model, clr in zip(range(len(self.modules)), self.modules, self.color):
            label = type(model).__name__
            plt.plot(self.train_loss[i, 40:], linewidth=1.0, color=clr, linestyle='--', label=label + " train")
            plt.plot(self.test_loss[i, 40:], linewidth=1.0, color=clr, label=label + " test")
            # plt.legend(prop={'size': 12})
            plt.xlabel("epoch")
            plt.title('Loss')

        plt.subplot(224)
        for i, model, clr in zip(range(len(self.modules)), self.modules, self.color):
            label = type(model).__name__
            plt.plot(self.test_acc[i, 40:], linewidth=1.0, color=clr, label=label + " test")
            # plt.legend(prop={'size': 12})
            plt.xlabel("epoch")
            plt.ylabel("%")
            plt.title('Test Accuarcy')
        plt.savefig('figure.svg', dpi=600, format='svg')
        plt.show()

    def __call__(self):

        if TRAIN:
            # 训练模型
            for i, model in zip(range(len(self.modules)), self.modules):
                print("====== " + type(model).__name__ + " ======")
                for epoch in range(1, EPOCH + 1):
                    self.epoch = epoch
                    self.train(i)
                    self.test(i)
                torch.save(self.modules[i].state_dict(), type(model).__name__ + ".pt")
            # 保存参数
            np.save('train_loss', self.train_loss)
            np.save('test_loss', self.test_loss)
            np.save('test_acc', self.test_acc)
        else:
            # 加载训练数据
            self.train_loss = np.load('train_loss.npy')
            self.test_loss = np.load('test_loss.npy')
            self.test_acc = np.load('test_acc.npy')
            # 加载模型参数
            """       
            for i, model in zip(range(len(self.modules)), self.modules):
                self.modules[i].load_state_dict(torch.load(type(model).__name__ + '.pt', map_location=self.device))
            """
        self.plot()

    def train(self, module_id):
        # 在训练集上训练
        self.modules[module_id].train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optim[module_id].zero_grad()
            output = self.modules[module_id](data)
            loss = self.criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            self.optim[module_id].step()
        train_loss /= 60000
        print('Epoch: ', self.epoch)

        print('Train set: Average loss: {:.4f}'.format(train_loss))
        self.train_loss[module_id][self.epoch - 1] = train_loss

    def test(self, module_id):
        # 在测试集上测试
        self.modules[module_id].eval()  # 模型进入eval模式，取消暂退
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.modules[module_id](data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= 10000
        accuracy = 100. * correct / 10000
        if TRAIN:
            self.test_loss[module_id][self.epoch - 1] = test_loss
            self.test_acc[module_id][self.epoch - 1] = accuracy
        print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, accuracy))
        print("------------")


if __name__ == "__main__":
    exp = MNISTexp()
    exp()
