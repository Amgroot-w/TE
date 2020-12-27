
"""
《北京化工大学学报》投稿用图
换一行，或者换一列试试效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from TE_plot import plot8

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def savefig(path):
    plt.savefig(path, bbox_inches='tight')  # 保存图片
    fig = Image.open(path)
    fig = fig.convert('L')  # 将RGB图片转为灰度图
    fig.save(path)  # 保存

def plot1():
    train_ac = 100 * pd.read_csv('train_ac.csv', header=None).values
    test_ac = 100 * pd.read_csv('test_ac.csv', header=None).values
    runtime = np.array([[46.253, 48.659, 50.61, 52.163, 78.024]])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax1 = plt.subplots()
    ax1.set_xticklabels([i for i in range(0, 22, 2)], fontsize='15')
    ax1.set_xticks([i for i in range(0, 22, 2)])
    ax1.tick_params(labelsize=15)
    plt.plot([2, 4, 5, 10, 20], train_ac[:, 1], 'o-r', markersize=9, alpha=1, label="训练集正确率")
    plt.plot([2, 4, 5, 10, 20], test_ac[:, 1], 's-', c='orange', markersize=9, alpha=1, label="测试集正确率")

    plt.xlabel('隐含层数', fontsize=15)
    plt.ylabel('正确率（%）', fontsize=15)
    plt.ylim(92, 98)
    plt.legend(loc='upper left', fontsize=14).get_frame().set_linewidth(0.0)

    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=15)
    plt.plot([2, 4, 5, 10, 20], runtime[0, :], '^-k', markersize=12, alpha=0.7, label='运行时间')
    ax2.set_ylabel('运行时间(s)', fontsize=15)
    plt.ylim(45, 84)
    plt.legend(loc='upper right', fontsize=14).get_frame().set_linewidth(0.0)
    savefig(r'PaperFigs\fig1.png')
    plt.show()

def plot2():
    train_ac = 100 * pd.read_csv('train_ac.csv', header=None).values
    test_ac = 100 * pd.read_csv('test_ac.csv', header=None).values
    runtime = np.array([[46.211, 46.981, 49.29, 50.087, 51.697]])
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    fig, ax1 = plt.subplots()
    ax1.set_xticklabels([16, 18, 20, 22, 24], fontsize='15')
    ax1.set_xticks([16, 18, 20, 22, 24])
    ax1.tick_params(labelsize=15)
    plt.plot([16, 18, 20, 22, 24], train_ac[3, :], 'o-r', markersize=9, alpha=1, label="训练集正确率")
    plt.plot([16, 18, 20, 22, 24], test_ac[3, :], 's-', c='orange', markersize=9, alpha=1, label="测试集正确率")

    plt.xlabel('隐层节点数', fontsize=15)
    plt.ylabel('正确率（%）', fontsize=15)
    plt.ylim(91, 99)
    plt.legend(loc='upper left', fontsize=14).get_frame().set_linewidth(0.0)

    ax2 = ax1.twinx()
    ax2.tick_params(labelsize=15)
    plt.plot([16, 18, 20, 22, 24], runtime[0, :], '^-k', markersize=12, alpha=0.7, label='运行时间')
    ax2.set_ylabel('运行时间(s)', fontsize=15)
    plt.ylim(46, 53)
    plt.legend(loc='upper right', fontsize=14).get_frame().set_linewidth(0.0)
    savefig(r'PaperFigs\fig2.png')
    plt.show()

def plot3():
    runtime = [22.916, 10.02, 6.024, 11.712, 52.328, 48.366]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.bar(range(6), runtime, width=0.6, fc='gray')
    plt.plot(range(6), runtime, 'o--k')
    plt.xticks(range(6), ['BP', 'SVM', 'PCA', 'KPCA', 'DAE', 'LSTM-DAE'], fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('模型', fontsize=15)
    plt.ylabel('运行时间 (s)', fontsize=15)
    plt.title('六类模型的运行时间', fontsize=15)
    savefig(r'PaperFigs\fig3.png')
    plt.show()

def plot4():
    bp = [64.31, 57.14, 66.27, 70.92, 69.03]
    svm = [63.59, 63.59, 63.59, 63.59, 63.59]
    pca = [85.00, 83.56, 89.31, 84.27, 89.44]
    kpca = [93.39, 90.17, 88.37, 89.95, 89.84]
    dae = [96.20, 96.13, 95.85, 96.11, 93.42]
    lstm = [97.24, 97.14, 97.05, 97.66, 97.48]
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(5), bp, '>--', label='BP', markersize=12)
    plt.plot(range(5), svm, '^--', label='SVM', markersize=12)
    plt.plot(range(5), pca, 's--c', label='PCA', markersize=12)
    plt.plot(range(5), kpca, 'D--y', label='KPCA', markersize=11)
    plt.plot(range(5), dae, 'o--g', label='DAE', markersize=12)
    plt.plot(range(5), lstm, '*--r', label='LSTM-DAE', markersize=15)

    plt.title('六类模型的正确率', fontsize=15)
    plt.xticks(range(5), [1, 2, 3, 4, 5], fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('实验次数', fontsize=15)
    plt.ylabel('正确率（%）', fontsize=15)
    plt.xlim(-0.5, 4.5)
    plt.ylim(15, 105)
    plt.legend(fontsize=13).get_frame().set_linewidth(0.0)
    savefig(r'PaperFigs\fig4.png')
    plt.show()

def plot5():
    for i in np.arange(1, 5):
        path = r'PaperFigs\图片%d.png' % i
        fig = Image.open(path)
        fig = fig.convert('L')  # 将RGB图片转为灰度图
        fig.save(path)  # 保存


if __name__ == '__main__':
    plot1()
    plot2()
    plot3()
    plot4()
    plot5()



