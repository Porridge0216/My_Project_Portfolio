import numpy as np
import random
import matplotlib.pyplot as plt
from math import sqrt, acos

E = 2.71828
PI = acos(-1)
EPS = 1e-6
TOE = 1e-3


class GmmExperienment:
    def __init__(self, data_num_, pi_, mu1_, sigma1_, mu2_, sigma2_, max_round_):

        self.data_num = data_num_
        self.pi = pi_
        self.mu1 = mu1_
        self.mu2 = mu2_
        self.sigma1 = sigma1_
        self.sigma2 = sigma2_

        self.pi_init = None
        self.mu1_init = None
        self.mu2_init = None
        self.sigma1_init = None
        self.sigma2_init = None
        self.random_init()

        self.max_round = max_round_
        self.round_fact = None
        self.round_fact_max = 0

        self.data = np.zeros((data_num_,), dtype="int")

        # 数据属于每个高斯分布的后验概率
        self.p_class1 = np.zeros((data_num_,), dtype="double")
        self.p_class2 = np.zeros((data_num_,), dtype="double")

        # 对数似然和迭代次数之间的关系
        self.log_like = []

        # 画图
        self.fig = plt.figure(figsize=(10, 4))
        self.ax = self.fig.add_subplot(121)
        self.bx = self.fig.add_subplot(122)

    def generate_data(self):
        """
        自动生成数据，通过随机数选择某个高斯分布
        :return:
        """
        for n in range(self.data_num):
            if random.randint(1, 100) <= self.pi * 100:
                self.generate_1(n)
            else:
                self.generate_2(n)

    def generate_1(self, index):
        """
        生成女生
        :param index: 生成的数据索引
        :return:
        """
        self.data[index] = np.random.normal(loc=self.mu1, scale=self.sigma1)

    def generate_2(self, index):
        """
        生成男生
        :param index: 生成的数据索引
        :return:
        """
        self.data[index] = np.random.normal(loc=self.mu2, scale=self.sigma2)

    def plot_hist(self):
        """
        画数据的直方图，间隔是1cm
        :return:
        """
        d = 1
        num_bins = int((max(self.data) - min(self.data)) / d)

        self.ax.hist(self.data, num_bins, alpha=0.5, color="gray")

    def plot_con(self, pi_, mu1_, sigma1_, mu2_, sigma2_, set_label):
        """
        绘制拟合曲线
        """

        x = np.arange(0, 500)
        con = (pi_ * self.n(x, mu1_, sigma1_) + (1 - pi_) * self.n(x, mu2_, sigma2_)) * self.data_num

        self.ax.plot(con, label=set_label)
        self.ax.legend(loc=0)
        self.ax.set(xlim=[min(self.data), max(self.data)], title='fit',
                    ylabel='count', xlabel='height')

        # plt.show()

    def plot_likelihood(self, set_label):
        """
        绘制似然函数随迭代次数的变化
        """
        self.bx.plot(self.log_like, label=set_label)
        self.bx.legend(loc=0)
        self.bx.set(title='log_likelihood',
                    xlabel='round', xlim=[0, self.round_fact_max])

    def em(self):
        """
        em算法，迭代e_step和m_step，最大化似然函数，返回估计的参数
        :return: 最终估计得到的参数
        """
        pi_new, mu1_new, mu2_new, sigma1_new, sigma2_new = self.pi_init, self.mu1_init, self.mu2_init, self.sigma1_init, self.sigma2_init
        for n in range(self.max_round):
            self.e_step(pi_new, mu1_new, sigma1_new, mu2_new, sigma2_new)
            pi_new, mu1_new, sigma1_new, mu2_new, sigma2_new = self.m_step()
            self.log_likelihood(pi_new, mu1_new, sigma1_new, mu2_new, sigma2_new)
            if (n > 0) and np.abs(self.log_like[-1] - self.log_like[-2]) < TOE:
                self.round_fact = n
                print("中断轮数", n)
                print("对数似然:", self.log_like[-1])
                if n > self.round_fact_max:
                    self.round_fact_max = n
                break

        return pi_new, mu1_new, sigma1_new, mu2_new, sigma2_new, self.log_like[-1]

    def e_step(self, pi_, mu1_, sigma1_, mu2_, sigma2_):
        # 计算Q函数，即参数给定的情况下，隐变量pi的条件概率
        self.p_class1 = (pi_ * self.n(self.data, mu1_, sigma1_)) / (pi_ * self.n(self.data, mu1_, sigma1_) + (1 - pi_) * self.n(self.data, mu2_, sigma2_))
        self.p_class2 = ((1 - pi_) * self.n(self.data, mu2_, sigma2_)) / (pi_ * self.n(self.data, mu1_, sigma1_) + (1 - pi_) * self.n(self.data, mu2_, sigma2_))
        return

    def m_step(self):
        pi_new = self.p_class1.sum() / self.data_num
        mu_1_new = (self.p_class1 * self.data).sum() / self.p_class1.sum()
        mu_2_new = (self.p_class2 * self.data).sum() / self.p_class2.sum()

        sigma_1_new = sqrt((self.p_class1 * (self.data - mu_1_new) * (self.data - mu_1_new)).sum() / self.p_class1.sum() + EPS)
        sigma_2_new = sqrt((self.p_class2 * (self.data - mu_2_new) * (self.data - mu_2_new)).sum() / self.p_class2.sum() + EPS)

        return pi_new, mu_1_new, sigma_1_new, mu_2_new, sigma_2_new

    def log_likelihood(self, pi_, mu1_, sigma1_, mu2_, sigma2_):
        """
        对数似然函数
        """
        self.log_like.append(np.log(pi_ * self.n(self.data, mu1_, sigma1_) + (1 - pi_) * self.n(self.data, mu2_, sigma2_)).sum())
        return

    @staticmethod
    def n(value, mu, sigma):
        """
        高斯分布
        """
        if 2 * sigma ** 2 <= 0.0001:
            sigma = 0.1

        return (1 / (sqrt(2 * PI) * sigma)) * E ** (-((value - mu) ** 2) / (2 * sigma ** 2))

    def random_init(self):
        """
        随机参数初始化
        """
        self.pi_init = random.random()
        self.mu1_init = random.randint(100, 250)
        self.mu2_init = random.randint(100, 250)
        self.sigma1_init = random.randint(1, 15)
        self.sigma2_init = random.randint(1, 15)


if __name__ == "__main__":
    # 设置数据生成器参数

    data_num = 1000
    pi = 0.45
    mu1 = 160
    mu2 = 178
    sigma1 = 5
    sigma2 = 15
    round_max = 1000

    # 构建实验
    exp1 = GmmExperienment(data_num, pi, mu1, sigma1, mu2, sigma2, round_max)

    # 生成数据
    exp1.generate_data()
    np.save("data.npy", exp1.data)
    print(exp1.data)
    # 画直方图
    exp1.plot_hist()

    # 设置实验次数
    exp_num = 10

    result = []
    likelihood_list = []
    for i in range(exp_num):
        print("实验" + str(i))
        # 随机初始化
        exp1.random_init()
        # 重置似然
        exp1.log_like = []
        # 拟合
        pi_e, mu1_e, sigma1_e, mu2_e, sigma2_e, likelihood = exp1.em()
        if likelihood is np.nan:
            likelihood = -999999999
        likelihood_list.append(likelihood)
        result.append([pi_e, mu1_e, sigma1_e, mu2_e, sigma2_e, likelihood])
        print("初始化参数" + str(i) + ":")
        print("pi:", exp1.pi_init, "mu1:", exp1.mu1_init, "sigma1:", exp1.sigma1_init, "mu2:", exp1.mu2_init, "sigma2:", exp1.sigma2_init)
        print("拟合结果" + str(i) + ":")
        print("pi:", pi_e, "mu1:", mu1_e, "sigma1:", sigma1_e, "mu2:", mu2_e, "sigma2:", sigma2_e)
        # 画拟合曲线
        exp1.plot_con(pi_e, mu1_e, sigma1_e, mu2_e, sigma2_e, set_label=str(i))
        # 画似然函数变化情况
        exp1.plot_likelihood(set_label=str(i))
        print("--------------------------")
    plt.show()

    best = result[likelihood_list.index(max(likelihood_list))]
    print("最优结果：")
    print(best)
