# ********************* TE_plot.py **********************
"""
实现功能：画图！

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot1():
    """
    绘制准确度ac随着类别数n的增加的变化曲线
    """
    x = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    ac_bp = {
        'train': [100, 99.5, 82.8, 81.9, 86, 80.3, 73, 69.6, 57.9, 55.4, 50, 47.46, 45.9],
        'test': [96, 96.6, 66.8, 64.8, 65, 54.8, 38.6, 41.3, 33.3, 29.1, 26.1, 24, 21.8]
    }
    ac_dae = {
        'train': [99, 99.3, 76.4, 68.4, 66, 62, 50.6, 38.3, 37.34, 28.1, 27.8, 24.1, 24.4],
        'test': [92, 89.5, 62.8, 45.54, 49, 33, 28.3, 25.2, 21.25, 22.6, 18.5, 15.5, 13.02]
    }

    plt.plot(x, ac_bp['train'], 'o-r', x, ac_dae['train'], 'o-b')
    plt.xlabel("n")
    plt.ylabel("Accuracy(%)")
    plt.title("train data")
    plt.show()

    plt.plot(x, ac_bp['test'], 'o-r', x, ac_dae['test'], 'o-b')
    plt.xlabel("n")
    plt.ylabel("Accuracy(%)")
    plt.title("test data")
    plt.show()

def plot2():
    """
    绘制变量随时间的变化曲线
    """
    # 原始特征（52维）随时间变化曲线
    train_data = pd.read_csv('D:\\Python Codes\\TE\\TE_Train_data.csv')  # 训练样本
    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)  # 归一化
    data = train_data[:500, :52]  # 选第一类正常工况数据
    plt.plot(range(500), data)
    plt.show()

    # 降维后的特征（20维）随时间变化曲线
    data1 = pd.read_csv('train_feature.csv', header=None).values
    plt.plot(range(980), data1)
    plt.show()

def plot3():
    """
    可视化每一类故障的准确率（训练、测试）
    测试集准确率低于80%的10个故障类别：3, 9, 10, 11, 13, 15, 16, 19, 21
    """
    train_ac0 = pd.read_csv('accuracy_train.csv', header=None).values[:, 1:]
    test_ac0 = pd.read_csv('accuracy_test.csv', header=None).values[:, 1:]
    train_ac = np.mean(train_ac0, axis=1)
    test_ac = np.mean(test_ac0, axis=1)

    name_list = ['Fault 1', 'Fault 2', 'Fault 3', 'Fault 4',
                 'Fault 5', 'Fault 6', 'Fault 7', 'Fault 8',
                 'Fault 9', 'Fault 10', 'Fault 11', 'Fault 12',
                 'Fault 13', 'Fault 14', 'Fault 15', 'Fault 16',
                 'Fault 17', 'Fault 18', 'Fault 19', 'Fault 20',
                 'Fault 21']

    width = 0.75  # 柱子的宽度
    # x1==x2时，合在一起显示，想要分开显示就让x2=x2+width
    x1 = np.array(range(len(train_ac)))
    x2 = np.array(range(len(test_ac)))
    x = np.array([-0.35, 20.5])

    # 画柱状图
    plt.bar(x1, train_ac, width=width, label='trian', fc='r')
    plt.bar(x2, test_ac, width=width, label='test', fc='orange')
    plt.plot(x, np.ones(x.shape), '--k', label='100%')
    plt.plot(x, 0.8*np.ones(x.shape), '--g', label='80%')

    plt.xticks(x1, name_list, rotation=60)  # 调节横坐标的倾斜度，rotation是度数
    plt.title('Acuracy of all 21 faults')
    plt.ylabel('Acuracy')
    plt.legend()  # 显示注释
    plt.show()

def plot4():
    name_list = ['16', '18', '20', '22', '24']
    train_ac = np.array([90.28, 90.34, 93.97, 94.10, 94.00])
    test_ac = np.array([92.93, 92.85, 96.08, 96.18, 96.09])
    average_ac = (train_ac + test_ac) / 2

    width = 0.3  # 柱子的宽度
    # x1==x2时，合在一起显示，想要分开显示就让x2=x2+width
    x1 = np.array(range(5))
    x2 = np.array(range(5)) + width
    x = (x1 + x2) / 2

    # 画柱状图
    plt.bar(x1, train_ac, width=width, label='训练集', fc='red')
    plt.bar(x2, test_ac, width=width, label='测试集', fc='orange')
    plt.plot(x, average_ac, 'o--k', label='平均值')
    for a, b in zip(x, average_ac):
        plt.text(a, b + 2, '%.2f%s' % (b, '%'), ha='center', va='bottom', fontsize=15)

    plt.xticks(x, name_list)  # 调节横坐标的倾斜度，rotation是度数
    plt.title('正确率随隐层节点数的变化曲线：DAE', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('正确率（%）', fontsize=20)
    plt.xlabel('隐层节点数', fontsize=20)
    plt.ylim(0, 107)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

def plot5():
    name_list = ['20', '25', '30', '32', '34']
    train_ac = np.array([80.21, 93.90, 94.10, 93.28, 94.14])
    test_ac = np.array([79.25, 96.01, 96.16, 95.36, 96.11])
    average_ac = (train_ac + test_ac) / 2

    width = 0.3  # 柱子的宽度
    # x1==x2时，合在一起显示，想要分开显示就让x2=x2+width
    x1 = np.array(range(5))
    x2 = np.array(range(5)) + width
    x = (x1 + x2) / 2

    # 画柱状图
    plt.bar(x1, train_ac, width=width, label='训练集', fc='red')
    plt.bar(x2, test_ac, width=width, label='测试集', fc='orange')
    plt.plot(x, average_ac, 'o--k', label='平均值')
    for a, b in zip(x, average_ac):
        plt.text(a, b + 2, '%.2f%s' % (b, '%'), ha='center', va='bottom', fontsize=15)

    plt.xticks(x, name_list)  # 调节横坐标的倾斜度，rotation是度数
    plt.title('正确率随隐层节点数的变化曲线：Softmax', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('正确率（%）', fontsize=20)
    plt.xlabel('隐层节点数', fontsize=20)
    plt.ylim(0, 107)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

def plot6():
    bp = [64.31, 57.14, 66.27, 70.92, 69.03]
    svm = [63.59, 63.59, 63.59, 63.59, 63.59]
    pca = [85.00, 83.56, 89.31, 84.27, 89.44]
    kpca = [93.39, 90.17, 88.37, 89.95, 89.84]
    dae = [96.20, 96.13, 95.85, 96.11, 93.42]
    plt.plot(range(5), bp, '>--', label='BP', markersize=12)
    plt.plot(range(5), svm, '^--', label='SVM', markersize=12)
    plt.plot(range(5), pca, 's--', label='PCA', markersize=12)
    plt.plot(range(5), kpca, 'D--', label='KPCA', markersize=11)
    plt.plot(range(5), dae, '*--', label='DAE', markersize=15)

    plt.title('不同模型的正确率', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('实验次数', fontsize=15)
    plt.ylabel('正确率（%）', fontsize=15)
    plt.xlim(-0.5, 4.5)
    plt.ylim(0, 107)
    plt.legend(fontsize=14)
    plt.show()

def plot7():
    ac = pd.read_csv('plot_ac.csv', header=None).values
    ac = 100 * ac

    namelist1 = ['16', '18', '20', '22', '24']  # 隐层节点数
    namelist2 = ['2', '4', '5', '10', '20']  # 时间深度
    markers = ['>', '^', 's', '*', 'D']
    markersizes = [12, 12, 12, 16, 11]

    for i in range(5):
        plt.plot(range(5), ac[i, :], 'o--', label='时间深度为%s' % namelist2[i],
                 marker=markers[i], markersize=markersizes[i])

    plt.scatter(1, ac[3, 1], c='red', marker='*', s=500, alpha=0.6)
    plt.text(1, ac[3, 1]+0.6, '%.2f%s' % (ac[3, 1], '%'), c='red', ha='center', va='bottom', fontsize=15)

    plt.xticks(range(5), namelist1)
    plt.title('网格搜索最优超参数', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('隐层节点数', fontsize=15)
    plt.ylabel('正确率（%）', fontsize=15)
    plt.ylim(80, 100)
    plt.legend(loc='lower right', fontsize=14)
    plt.show()

def plot8():
    bp = [64.31, 57.14, 66.27, 70.92, 69.03]
    svm = [63.59, 63.59, 63.59, 63.59, 63.59]
    pca = [85.00, 83.56, 89.31, 84.27, 89.44]
    kpca = [93.39, 90.17, 88.37, 89.95, 89.84]
    dae = [96.20, 96.13, 95.85, 96.11, 93.42]
    lstm = [97.24, 97.14, 97.05, 97.66, 97.48]
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
    plt.legend(fontsize=14)
    plt.show()


if __name__ == '__main__':
    # plot1()
    # plot2()
    # plot3()
    # 以下为论文用图
    plt.figure()
    plot4()
    plt.figure()
    plot5()
    plt.figure()
    plot6()
    plt.figure()
    plot7()
    plt.figure()
    plot8()
