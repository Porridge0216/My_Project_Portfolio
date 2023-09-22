import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
import torch.optim as optim
import glob

from natsort import natsorted

import torch.nn as nn
import torch

from torch.nn import functional as F
from matplotlib import pyplot as plt

TRAIN_IMG_DIR = "./trainset/input/*.png"
TRAIN_LAB_DIR = "./trainset/label/*.png"

TEST_IMG_DIR = "./testset/input/*.png"
TEST_LAB_DIR = "./testset/label/*.png"

BATCH_SIZE = 1
EPOCH = 50
TRAIN = False


# 若为False，要确保当前目录下存在模型的pt文件和训练数据npy文件
# 搭建unet 网络
class DoubleConv(nn.Module):  # 连续两次卷积
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # 用 BN 代替 Dropout
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):  # 下采样
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.downsampling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.downsampling(x)
        return x


class Up(nn.Module):  # 上采样
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.upsampling = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 转置卷积
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsampling(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])  # 确保任意size的图像输入
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)  # 从channel 通道拼接
        x = self.conv(x)
        return x


class OutConv(nn.Module):  # 最后一个网络的输出
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):  # unet 网络
    def __init__(self, in_channels=1, num_classes=1):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.in_conv = DoubleConv(in_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.out_conv = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)

        return x


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


class Exp:
    def __init__(self):
        self.test_loader = None
        self.train_loader = None

        self.train_size = None
        self.test_size = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 制作tensor数据集，生成dataloader
        self.make_dataset()

        # 实例化模型
        self.module = UNet(3, 1).to(self.device)
        # 交叉熵损失
        # self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        # 训练数据记录
        self.train_loss = np.zeros(EPOCH)
        self.test_loss = np.zeros_like(self.train_loss)

        # 参数量分析，生成优化器
        print("====== " + type(self.module).__name__ + " ======")
        # torchsummary.summary(self.module, (1, 512, 512))
        # 随机梯度下降优化器
        self.optim = optim.RMSprop(self.module.parameters(), lr=0.00001, weight_decay=1e-8, momentum=0.9)

        self.epoch = 0
        self.best_epoch = 0
        self.best_test_loss = float('inf')

        """
        用于绘制ROC曲线
        """
        self.target = []
        self.output = []

        self.confuse_matrix = []  # TP FN; FP TN
        self.tpr = []  # 真正例率  TP/(TP+FN)
        self.fpr = []  # 假正例率  FP/(TN+FP)

    def make_dataset(self):
        # 读数据集
        files_img_train = glob.glob(TRAIN_IMG_DIR)
        files_img_train = natsorted(files_img_train)

        files_lab_train = glob.glob(TRAIN_LAB_DIR)
        files_lab_train = natsorted(files_lab_train)

        files_img_test = glob.glob(TEST_IMG_DIR)
        files_img_test = natsorted(files_img_test)


        files_lab_test = glob.glob(TEST_LAB_DIR)
        files_lab_test = natsorted(files_lab_test)

        self.train_size = len(files_img_train)
        self.test_size = len(files_img_test)

        train_data = torch.zeros((len(files_img_train), 3, 584, 565))
        train_label = torch.zeros((len(files_lab_train), 1, 584, 565))

        test_data = torch.zeros((len(files_img_test), 3, 584, 565))

        test_label = torch.zeros((len(files_lab_test), 1, 584, 565))

        transform1 = transforms.Compose(
            [transforms.Normalize(mean=[129.5006, 68.3902, 41.1953], std=[86.1404, 44.7121, 24.9279])])

        transform2 = transforms.Compose([transforms.Grayscale()])

        for file,  i in zip(files_img_train,  range(len(files_img_train))):
            train_data[i] = transform1(read_image(file).float())

        for file, i in zip(files_lab_train, range(len(files_lab_train))):
            train_label[i] = transform2(read_image(file).float()) / 255

        for file,  i in zip(files_img_test,  range(len(files_img_test))):
            test_data[i] = transform1(read_image(file).float())

        for file, i in zip(files_lab_test, range(len(files_lab_test))):
            test_label[i] = transform2(read_image(file).float()) / 255

        train_set = EXPDataset(train_data, train_label)
        test_set = EXPDataset(test_data, test_label)

        self.train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

        """
        mean = torch.zeros(3)
        std = torch.zeros(3)
        total_images = 0


        for images, _ in self.train_loader:
            batch_size = images.size(0)
            images = images.view(batch_size, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_size

        for images, _ in self.test_loader:
            batch_size = images.size(0)
            images = images.view(batch_size, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_size

        mean /= total_images
        std /= total_images

        print("Mean: ", mean)
        print("Std: ", std)
        """

    def plot(self):
        """
        绘制Loss和epoch关系
        """
        plt.figure(figsize=(10, 7))
        plt.plot(self.train_loss, linewidth=1.0, linestyle='--', label="train")
        plt.plot(self.test_loss, linewidth=1.0, label="test")
        plt.legend()
        plt.xlabel("epoch")
        plt.title('Loss')

        plt.savefig('loss.svg', dpi=600, format='svg')
        plt.show()

    def roc(self):
        """
        绘制ROC曲线
        """
        print("plot...")
        thres = []
        acc = []
        tpr = []
        tnr = []

        thress = 0
        while thress <= 1000:
            threshold = thress / 1000
            thres.append(threshold)

            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for tar, out in zip(self.target, self.output):
                # 真正例
                tp += ((out > threshold) * tar.bool()).sum().item()
                # 真反例
                tn += ((out <= threshold) * (~(tar.bool()))).sum().item()
                # 假正例
                fp += ((out > threshold) * (~(tar.bool()))).sum().item()
                # 假反例
                fn += ((out <= threshold) * tar.bool()).sum().item()

            self.tpr.append(tp / (tp + fn))
            self.fpr.append(fp / (tn + fp))
            thress += 1
            # 准确率
            acc.append((tp + tn) / (tp + tn + fp + fn))

            # 敏感度
            tpr.append(tp / (tp + fn))
            # 特异性 真阴性率  特异性 = (真阴性的样本数) / (实际为负例的样本数)
            tnr.append(tn / (fp + tn))

        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        plt.plot(self.fpr, self.tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")

        plt.title("ROC")

        plt.subplot(122)
        plt.plot(thres, acc, label="ACC")
        plt.plot(thres, tpr, label="TPR")
        plt.plot(thres, tnr, label="TNR")
        plt.xlabel("threshold")
        plt.legend()
        plt.savefig('roc.svg', dpi=600, format='svg')
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

        else:
            # 加载训练数据
            try:
                self.train_loss = np.load('train_loss.npy')
                self.test_loss = np.load('test_loss.npy')
            except FileNotFoundError:
                print("没有找到训练记录npy文件")
            try:
                self.module.load_state_dict(torch.load(type(self.module).__name__ + '.pt', map_location=self.device))
                self.test()
            except FileNotFoundError:
                print("没有找到模型参数字典pt文件")

            self.plot()
            self.roc()

    @staticmethod
    def dice_loss(pred, target, smooth=1e-5):
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss

    def train(self):
        # 在训练集上训练
        self.module.train()
        train_loss = 0.0
        i = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            i += 1
            # print(i)

            data = data.to(self.device)
            target = target.to(self.device)
            self.optim.zero_grad()
            output = self.module(data)
            # target = target.squeeze(dim=1)
            loss = self.criterion(output, target)
            # loss = self.dice_loss(torch.sigmoid(output), target)

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
        i = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                i += 1
                # print(i)

                data = data.to(self.device)
                target = target.to(self.device)
                output = self.module(data)
                # target = target.squeeze(dim=1)

                test_loss += self.criterion(output, target).item()
                # test_loss += self.dice_loss(torch.sigmoid(output), target).item()

                self.target.append(target.squeeze(dim=1).squeeze(dim=0))
                self.output.append(torch.sigmoid(output.squeeze(dim=1).squeeze(dim=0)))

                # 绘制测试样例

                if not TRAIN:
                    plt.figure()
                    plt.subplot(131)
                    plt.imshow(data[0][0].cpu())
                    plt.title("input")

                    plt.subplot(132)
                    plt.imshow(target[0][0].cpu())
                    plt.title("target")

                    plt.subplot(133)
                    plt.imshow(torch.sigmoid(output[0][0]).cpu())
                    plt.title("output")

                    plt.savefig(str(i) + ".png")

        test_loss /= self.test_size

        if TRAIN:
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                torch.save(self.module.state_dict(), type(self.module).__name__ + ".pt")
                self.best_epoch = self.epoch

        if TRAIN:
            self.test_loss[self.epoch - 1] = test_loss

        print('Test set: Average loss: {:.4f}'.format(test_loss))
        print("------------")


if __name__ == "__main__":
    exp = Exp()
    exp()
