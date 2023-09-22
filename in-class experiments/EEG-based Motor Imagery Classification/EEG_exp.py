import glob
import os
import sys
import mne
import scipy
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchsummary
import torch.nn as nn
import cv2
import numpy as np
import warnings

BATCH_SIZE = 128
EPOCH = 500
CLASS = 4  # 2 or 4
LR = 0.01
# 用谁的数据训练，用谁的数据评估。*代表合一起
TRAIN_FROM = 3  # 1-9  or '*'
APPLY_TO = 3  # 1-9 or '*'
EVALUATE_ONLY = True  # False代表既训练又评估

PT_SAVE_DIR = 'best_' + (str(TRAIN_FROM) if TRAIN_FROM != '*' else 'all') + '_cls_' + str(CLASS) + '.pt'
PT_READ_DIR = 'best_' + (str(TRAIN_FROM) if TRAIN_FROM != '*' else 'all') + '_cls_' + str(CLASS) + '.pt'

C = 22
T = 751


class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        N = CLASS
        F1 = 8
        D = 2
        F2 = 16

        # 1 输入为 C，T
        # 2 增加维度，变成 1，C，T
        # 3 普通卷积，1通道变F1通道，卷积核 1，64 same模式
        self.conv3 = nn.Conv2d(1, F1, (1, 64), padding='same', bias=False)
        # 4 batchNorm
        self.batchnorm4 = nn.BatchNorm2d(F1)
        # 5 深度卷积，8通道变2*8通道 卷积核22，1  valid模式，maxNorm=1
        self.depthwiseConv5 = nn.Conv2d(F1, D * F1, (C, 1), padding='valid', groups=F1, bias=False)
        nn.utils.clip_grad_norm(self.depthwiseConv5.parameters(), max_norm=1)
        # 6 batchNorm
        self.batchnorm6 = nn.BatchNorm2d(D * F1)
        # 7 activation  ELU
        self.elu7 = nn.ELU()
        # 8 平均池化 1，4
        self.pool8 = nn.AvgPool2d((1, 4))
        # 9 dropout 0.5
        self.dropout9 = nn.Dropout(0.5)
        # 10 分离卷积 same  先深度卷积，再pointwise
        self.depthwise_conv10 = nn.Conv2d(D * F1, F2, (1, 16), padding='same', groups=D * F1, bias=False)
        self.pointwise_conv10 = nn.Conv2d(F2, F2, kernel_size=1, bias=False)
        # 11 batchNorm
        self.batchnorm11 = nn.BatchNorm2d(F2)
        # 12 ELU激活
        self.elu12 = nn.ELU()
        # 13 平均池化 1，8
        self.pool13 = nn.AvgPool2d((1, 8))
        # 14 dropout 0.5
        self.dropout14 = nn.Dropout(0.5)
        # 15 拉直
        self.flatten15 = nn.Flatten()
        # 16 softmax四分类
        self.dense16 = nn.Linear(F2 * (T // 32), N)
        nn.utils.clip_grad_norm(self.dense16.parameters(), max_norm=0.25)

    def forward(self, x):
        # 1 输入，(batch) C T 下文的batch都省略
        x = x.unsqueeze(1)  # 2 在通道维度上增加一维  1 C T
        # 3 same模式，conv2d
        x = self.conv3(x)  # F1 C T
        # 4 batchnorm
        x = self.batchnorm4(x)
        # 5 深度卷积 valid模式，max_norm = 1
        x = self.depthwiseConv5(x)  # D*F1 1 T
        # 6 batchnorm
        x = self.batchnorm6(x)
        # 7 ELU激活
        x = self.elu7(x)
        # 8 平均池化 1 4
        x = self.pool8(x)  # D*F1 1 T//4
        # 9 dropout
        x = self.dropout9(x)
        # 10 分离卷积 same
        x = self.depthwise_conv10(x)
        x = self.pointwise_conv10(x)
        # 11 batchnorm
        x = self.batchnorm11(x)
        # 12 ELU
        x = self.elu12(x)
        # 13 平均池化 1 8
        x = self.pool13(x)
        # 14 dropout
        x = self.dropout14(x)
        # 15 Flatten层
        x = self.flatten15(x)  # B F2*T//2
        # 16 全连接层
        x = self.dense16(x)
        x = nn.functional.softmax(x, dim=1)
        return x


class EEGExperiment:
    def __init__(self):
        self.files_train = None  # 训练数据文件
        self.files_evalu = None  # 评估数据文件
        self.labels_files_evalu = None  # 测试数据的真实标签

        self.dataset_list_train = []  # 预处理后的训练数据
        self.labelset_list_train = []  # 训练标签
        self.dataset_list_evalu = []  # 预处理后的测试数据
        self.labelset_list_evalu = []  # 测试数据的真实标签

        self.sample_data = None
        self.sample_label = None

        # torch数据集
        self.dataset_loader_train = None
        self.dataset_loader_val = None
        self.dataset_loader_test = None

        self.batch_train = 288
        self.shape_train = (22, 751)

        # 模型
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float('inf')
        self.loss_train = []
        self.loss_val = []
        self.acc_val = []

    def __call__(self):
        self.read_files()
        self.get_data()
        self.show_data()
        if not EVALUATE_ONLY:
            self.train_module()
            self.show_train_info()
        self.test()

    def read_files(self):
        self.files_train = glob.glob("./Data/*" + str(TRAIN_FROM) + "T.gdf")
        self.files_train = sorted(self.files_train, key=lambda f: f[-6], reverse=False)
        self.files_evalu = glob.glob("./Data/*" + str(APPLY_TO) + "E.gdf")
        self.files_evalu = sorted(self.files_evalu, key=lambda f: f[-6], reverse=False)
        self.labels_files_evalu = glob.glob("./Label/*" + str(APPLY_TO) + "E.mat")
        self.labels_files_evalu = sorted(self.labels_files_evalu, key=lambda f: f[-6], reverse=False)
        print("Files are set as follows:")
        print("train files: ", self.files_train)
        print("evaluate files: ", self.files_evalu)
        print("evaluate labels files; ", self.labels_files_evalu)
        return

    def validate(self, model, dataloader, crit, test=False):
        model.eval()
        running_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                outputs = model(inputs.to(self.device))
                loss = crit(outputs, labels.to(self.device))
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels.to(self.device)).sum().item()
            # scheduler.step(running_loss)
        if not test:
            self.loss_val.append(running_loss / len(dataloader))
            self.acc_val.append(total_correct / len(dataloader.dataset))
            if running_loss / len(dataloader) < self.best_val_loss:
                self.best_val_loss = running_loss / len(dataloader)
                torch.save(self.model.state_dict(), PT_SAVE_DIR)
        return running_loss / len(dataloader), total_correct / len(dataloader.dataset)

    def train_module(self):
        def train(model, dataloader, opt, crit):
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader, 0):
                outputs = model(inputs.to(self.device))
                loss = crit(outputs, labels.to(self.device))
                running_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()

            self.loss_train.append(running_loss / len(dataloader))
            return running_loss / len(dataloader)

        self.model = EEGNet().to(self.device).double()
        torchsummary.summary(EEGNet(), (22, 751))
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        # 开始训练
        for epoch in range(EPOCH):
            train_loss = train(self.model, self.dataset_loader_train, optimizer, criterion)
            val_loss, val_acc = self.validate(self.model, self.dataset_loader_val, criterion)

            print(f'Epoch {epoch + 1} -- Training Loss: {train_loss:.4f} -- Validation Loss: {val_loss:.4f} -- Validation Accuracy: {val_acc:.4f}')
            # print('Lr = ', optimizer.param_groups[0]['lr'])

    def show_train_info(self):
        plt.subplot(121)
        plt.plot(self.loss_train, label="train")
        plt.plot(self.loss_val, label="val")
        plt.legend(loc="upper left")
        plt.xlabel("epoch")
        plt.title('LOSS')
        plt.subplot(122)
        plt.plot(self.acc_val)
        plt.xlabel("epoch")
        plt.title('ACC')
        plt.savefig('train_info_cls' + str(CLASS) + '_tester_' + str(TRAIN_FROM) + '.png')
        plt.show()

    def test(self):
        self.model = EEGNet().to(self.device).double()
        self.model.load_state_dict(torch.load(PT_READ_DIR))
        torchsummary.summary(EEGNet(), (22, 751))
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = self.validate(self.model, self.dataset_loader_test, criterion, test=True)
        print("====== Test on " + str(APPLY_TO) + " ======")
        print(f'Test Loss: {val_loss:.4f} -- Test Accuracy: {val_acc:.4f}')
        print('FINISHED!')

    @staticmethod
    def pre_process_gdf(filepath, withlabel=True):
        mapping = {
            'EEG-Fz': 'eeg',
            'EEG-0': 'eeg',
            'EEG-1': 'eeg',
            'EEG-2': 'eeg',
            'EEG-3': 'eeg',
            'EEG-4': 'eeg',
            'EEG-5': 'eeg',
            'EEG-C3': 'eeg',
            'EEG-6': 'eeg',
            'EEG-Cz': 'eeg',
            'EEG-7': 'eeg',
            'EEG-C4': 'eeg',
            'EEG-8': 'eeg',
            'EEG-9': 'eeg',
            'EEG-10': 'eeg',
            'EEG-11': 'eeg',
            'EEG-12': 'eeg',
            'EEG-13': 'eeg',
            'EEG-14': 'eeg',
            'EEG-Pz': 'eeg',
            'EEG-15': 'eeg',
            'EEG-16': 'eeg',
            'EOG-left': 'eog',
            'EOG-central': 'eog',
            'EOG-right': 'eog'
        }
        raw = mne.io.read_raw_gdf(filepath)
        raw.set_channel_types(mapping)
        events, events_dict = mne.events_from_annotations(raw)
        raw.load_data()
        raw.plot()
        # 4到40Hz滤波
        raw.filter(4., 40., fir_design='firwin', n_jobs=6)
        # 通道选择，除去EOG通道
        picks = mne.pick_types(raw.info, eeg=True)
        # 提取cue时段内的数据
        tmin, tmax = 1., 4.
        # 左手 = 769,右手 = 770,脚 = 771,舌头 = 772
        event_names = ['769', '770', '771', '772'] if withlabel else ['783']
        event_id = {name: events_dict[name] for name in event_names}
        epoch = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                           baseline=None, preload=True)
        return epoch

    @staticmethod
    def pre_process_mat(filepath):
        raw_mat = scipy.io.loadmat(filepath)
        raw_data = raw_mat['classlabel']
        raw_data = raw_data.reshape(288, ) - 1
        return raw_data

    def get_data(self):
        # 得到训练数据和训练标签
        for filepath in self.files_train:
            sys.stdout = sys.__stdout__
            print("Extracting useful data from " + str(filepath) + " ...")
            sys.stdout = open(os.devnull, 'w')
            epoch = self.pre_process_gdf(filepath)
            labels = torch.from_numpy((epoch.events[:, -1] - epoch.events[:, -1].min()))  # 288, ndarray
            data = torch.from_numpy(epoch.get_data())  # 288*22*751 ndarray
            self.dataset_list_train.append(data)
            self.labelset_list_train.append(labels.long())
        # 得到评估数据和评估标签
        for filepath in self.files_evalu:
            sys.stdout = sys.__stdout__
            print("Extracting useful data from " + str(filepath) + " ...")
            sys.stdout = open(os.devnull, 'w')
            epoch = self.pre_process_gdf(filepath, False)
            data = torch.from_numpy(epoch.get_data())  # 288*22*751 ndarray
            self.dataset_list_evalu.append(data)
        sys.stdout = sys.__stdout__
        for filepath in self.labels_files_evalu:
            print("Extracting useful data from " + str(filepath) + " ...")
            label = torch.from_numpy(self.pre_process_mat(filepath))
            self.labelset_list_evalu.append(label.long())
        self.sample_data = self.dataset_list_train[0]
        self.sample_label = self.labelset_list_train[0]
        self.dataset_list_train = torch.vstack(self.dataset_list_train)
        self.labelset_list_train = torch.hstack(self.labelset_list_train)
        self.dataset_list_evalu = torch.vstack(self.dataset_list_evalu)
        self.labelset_list_evalu = torch.hstack(self.labelset_list_evalu)

        if CLASS == 2:
            # 筛选数据集
            self.dataset_list_train = [self.dataset_list_train[i] for i in range(len(self.labelset_list_train)) if
                                       (self.labelset_list_train[i] == 0 or self.labelset_list_train[i] == 1)]
            self.labelset_list_train = [self.labelset_list_train[i] for i in range(len(self.labelset_list_train)) if
                                        (self.labelset_list_train[i] == 0 or self.labelset_list_train[i] == 1)]
            self.dataset_list_evalu = [self.dataset_list_evalu[i] for i in range(len(self.labelset_list_evalu)) if
                                       (self.labelset_list_evalu[i] == 0 or self.labelset_list_evalu[i] == 1)]
            self.labelset_list_evalu = [self.labelset_list_evalu[i] for i in range(len(self.labelset_list_evalu)) if
                                        (self.labelset_list_evalu[i] == 0 or self.labelset_list_evalu[i] == 1)]

        dataset = EEGDataset(self.dataset_list_train, self.labelset_list_train)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        dataset_train, dataset_val = random_split(dataset, [train_size, val_size])
        dataset_test = EEGDataset(self.dataset_list_evalu, self.labelset_list_evalu)

        self.dataset_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
        self.dataset_loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
        self.dataset_loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    def show_data(self):
        index = []
        for i in range(4):
            index.append(np.where(self.sample_label == i)[0][0])
        # print(index)
        count = 1
        for i in range(4):
            plt.subplot(4, 2, count)
            img = self.sample_data[index[i]]
            img = np.float32(img)
            dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)  # 傅立叶变换
            dft_shift = np.fft.fftshift(dft)  # 调用shift方法将低频部分转移到中心位置
            # 将傅立叶变换结果映射到一个单通道二维数组中
            magnitude_specturm = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            plt.imshow(magnitude_specturm, aspect=5)
            plt.title(str(769 + i) + " magnitude specturm")

            plt.subplot(4, 2, count + 1)
            plt.imshow(img, aspect=5)
            plt.title(str(769 + i) + " raw data")
            count += 2
        plt.savefig('signal_examples.png')
        plt.show()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    exp = EEGExperiment()
    exp()
