
"""
《北京化工大学学报》投稿用图
换一行，或者换一列试试效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot1():
    train_ac = 100 * pd.read_csv('train_ac.csv', header=None).values
    test_ac = 100 * pd.read_csv('test_ac.csv', header=None).values
    runtime = np.array([[46.253, 48.659, 50.61, 52.163, 78.024]])

    fig, ax1 = plt.subplots()
    ax1.set_xticklabels([i for i in range(0, 22, 2)], fontsize='15')
    ax1.set_xticks([i for i in range(0, 22, 2)])
    ax1.tick_params(labelsize=15)
    plt.plot([2, 4, 5, 10, 20], train_ac[:, 1], 'o-r', markersize=9, alpha=1, label="训练集正确率")
    plt.plot([2, 4, 5, 10, 20], test_ac[:, 1], 's-', c='orange', markersize=9, alpha=1, label="测试集正确率")

    plt.xlabel('隐含层数', fontsize=17)
    plt.ylabel('正确率（%）', fontsize=17)
    plt.ylim(92, 98)
    plt.legend(loc='upper left', fontsize=14)

    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=15)
    plt.plot([2, 4, 5, 10, 20], runtime[0, :], '^-k', markersize=12, alpha=0.7, label='运行时间')
    ax2.set_ylabel('运行时间(s)', fontsize=17)
    plt.ylim(45, 84)
    plt.legend(loc='upper right', fontsize=14)

    plt.show()


def plot2():
    train_ac = 100 * pd.read_csv('train_ac.csv', header=None).values
    test_ac = 100 * pd.read_csv('test_ac.csv', header=None).values
    runtime = np.array([[46.211, 46.981, 49.29, 50.087, 51.697]])

    fig, ax1 = plt.subplots()
    ax1.set_xticklabels([16, 18, 20, 22, 24], fontsize='15')
    ax1.set_xticks([16, 18, 20, 22, 24])
    ax1.tick_params(labelsize=15)
    plt.plot([16, 18, 20, 22, 24], train_ac[3, :], 'o-r', markersize=9, alpha=1, label="训练集正确率")
    plt.plot([16, 18, 20, 22, 24], test_ac[3, :], 's-', c='orange', markersize=9, alpha=1, label="测试集正确率")

    plt.xlabel('隐层节点数', fontsize=17)
    plt.ylabel('正确率（%）', fontsize=17)
    plt.ylim(91, 99)
    plt.legend(loc='upper left', fontsize=14)

    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=15)
    plt.plot([16, 18, 20, 22, 24], runtime[0, :], '^-k', markersize=12, alpha=0.7, label='运行时间')
    ax2.set_ylabel('运行时间(s)', fontsize=17)
    plt.ylim(46, 53)
    plt.legend(loc='upper right', fontsize=14)

    plt.show()


def plot3():
    runtime = [22.916, 10.02, 6.024, 11.712, 52.328, 48.366]
    plt.bar(range(6), runtime, width=0.6, fc='gray')
    plt.plot(range(6), runtime, 'o--k')
    plt.xticks(range(6), ['BP', 'SVM', 'PCA', 'KPCA', 'DAE', 'LSTM-DAE'], fontsize=15)
    plt.xlabel('模型', fontsize=17)
    plt.ylabel('运行时间（s）', fontsize=17)
    plt.title('六类模型的运行时间', fontsize=17)
    plt.show()


if __name__ == '__main__':
    plot1()
    plot2()
    plot3()



