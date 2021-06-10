"""
                      CAP.py
                                Zhang Jingchuan, NEU, ISE, DAO
===============================================================
1. 算法：
Logistic回归：      cap.logistic(x, y, epochs, alpha, lamda)
K-means算法：       cap.kmeans(data, K, iteration)
PCA算法：           cap.pca(data)

---------------------------------------------------------------
2. 基础函数：
Sigmoid函数：        cap.sigmoid(x)
Sigmoid函数的导数：   cap.d_sigmoid(x)
tanh函数：           cap.tanh(x)
tanh函数的导数：      cap.d_tanh(x)
Softmax函数：        cap.softmax(x)
交叉熵损失函数：       cap.cross_entropy(x, y)
均方误差损失函数：     cap.mse(x, y)
计算Fβ-Score:       cap.F1_Score(pred_, true_, beta_)

---------------------------------------------------------------
3.数据预处理：
异常值处理：         cap.Outlier(dataframe, method)
特征归一化：         cap.normalize(features, method)
标签one-hot编码：    cap.onehot(labels)
读取MNIST数据集      cap.read_mnist(dataPath)
---------------------------------------------------------------
4.其他：
DataSet类：         cap.DataSet(images, labels)
多张图片可视化：       cap.display(images, pixel, size)
一维特征->高维特征：   cap.polyfeatures(x, degree)
二维特征->高维特征：   cap.feature_mapping(x1, x2, degree)
绘制原始样本分布:      cap.plot_original_data(data)
绘制决策边界:         cap.plot_decision_boundary(data, theta)

===============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip

# ===============================================================

# %% 读取MNIST数据集
def read_mnist(dataPath=r'D:\Python Codes\CNN\MNIST_data'):
    """
    :param dataPath: 默认路径为 D:\\Python Codes\\CNN\\MNIST_data
    :return: train_x: (60000, 784)
             train_y: (60000, 1)
             test_x: (10000, 784)
             test_y: (10000, 1)
    """
    def read_idx3(filename):
        with gzip.open(filename, 'rb') as fo:
            buf = fo.read()
            index = 0
            header = np.frombuffer(buf, '>i', 4, index)
            index += header.size * header.itemsize
            data = np.frombuffer(buf, '>B', header[1] * header[2] * header[3], index).reshape(header[1], -1)
            return data

    def read_idx1(filename):
        with gzip.open(filename, 'rb') as fo:
            buf = fo.read()
            index = 0
            header = np.frombuffer(buf, '>i', 2, index)
            index += header.size * header.itemsize
            data = np.frombuffer(buf, '>B', header[1], index)
            return data
        
    train_x = read_idx3(dataPath + '/train-images-idx3-ubyte.gz')  # 训练数据集的样本特征
    train_y = read_idx1(dataPath + '/train-labels-idx1-ubyte.gz')  # 训练数据集的标签
    test_x = read_idx3(dataPath + '/t10k-images-idx3-ubyte.gz')  # 测试数据集的样本特征
    test_y = read_idx1(dataPath + '/t10k-labels-idx1-ubyte.gz')  # 测试数据集的标签
    return train_x, train_y.reshape(-1, 1), test_x, test_y.reshape(-1, 1)


# %% tanh函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# %% tanh函数的导数
def d_tanh(x):
    return pow(1-tanh(x), 2)


# %% sigmoid 函数
# 输入x：数、向量、矩阵（均为对元素操作，不降维）
def sigmoid(x):
    np.seterr(divide='ignore', invalid='ignore')  # 去掉警告
    return 1 / (1 + np.exp(-x))


# %% sigmoid函数的导数
def d_sigmoid(x):
    return np.multiply(sigmoid(x), 1-sigmoid(x))


# %% Softmax函数
def softmax(x):
    res = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
    return res


# %% 计算Fβ-Score
# 默认β=1，即默认计算F1-Score
# 输入两个np数组（n×1）：预测值pred_，真实值true_
def F1_Score(pred_, true_, beta_=1):
    pred_ = np.array([0 if pred_[i] == 0 else 1 for i in range(len(pred_))]).reshape([-1, 1])
    true_ = np.array([0 if true_[i] == 0 else 1 for i in range(len(true_))]).reshape([-1, 1])
    TP = np.sum((true_ == 1) & (pred_ == 1))
    FP = np.sum((true_ == 0) & (pred_ == 1))
    FN = np.sum((true_ == 1) & (pred_ == 0))
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    res = (1+beta_**2)*P*R / (beta_**2*P+R)
    return res

# %% 计算误报率FAR、漏检率MDR
def FAR_MDR(pred_, true_):
    pred_ = np.array([0 if pred_[i] == 0 else 1 for i in range(len(pred_))]).reshape([-1, 1])
    true_ = np.array([0 if true_[i] == 0 else 1 for i in range(len(true_))]).reshape([-1, 1])
    TP = np.sum((true_ == 1) & (pred_ == 1))
    FP = np.sum((true_ == 0) & (pred_ == 1))
    FN = np.sum((true_ == 1) & (pred_ == 0))
    TN = np.sum((true_ == 0) & (pred_ == 0))
    FAR = FP / (FP + TN)
    MDR = FN / (TP + FN)
    return FAR, MDR

# %% one-hot编码
# **只支持二分类
# 输入：y为一个序列
# 输出：res为一个矩阵
def onehot(labels):
    res = np.zeros([labels.shape[0], int(np.max(labels)+1)])
    for i in range(labels.shape[0]):
        res[i, int(labels[i])] = 1
    return res


# %% 归一化
# 提供三种方法：MaxMin方法(默认)，Z-score方法，Sigmoid方法
def normalize(features, method='maxmin'):
    np.seterr(divide='ignore', invalid='ignore')  # 去掉警告
    if method == 'maxmin':
        maximum = np.max(features, axis=0)
        minimum = np.min(features, axis=0)
        res = (features - minimum) / (maximum - minimum)
        return res

    elif method == 'zscore':
        mu = np.mean(features, axis=0)
        sigma = np.std(features, axis=0)
        res = (features - mu) / sigma
        return res

    elif method == 'sigmoid':  # 双重循环实现，比较费时
        return 1 / (1 + np.exp(-features))

    else:
        print('****** 归一化函数的 method 输入错误！ ******')


# %% 均方误差损失函数
# 输入x, y可以是数、向量、矩阵，返回值均为一个数
def mse(x, y):
    return 1/2 * np.mean(np.sum(pow(x - y, 2), axis=1))


# %% 交叉熵损失函数
# 输入x, y可以是数、向量、矩阵，返回值均为一个数
def cross_entropy(x, y):
    np.seterr(divide='ignore', invalid='ignore')  # 去掉警告
    return np.mean(np.sum(-np.multiply(y, np.log(x))
                          - np.multiply((1-y), np.log(1-x)), axis=1))


# %% 多张图片可视化
def display(images, pixel, size):
    # images为图片集数据矩阵
    # pixel表示像素 (e.g. [32×32]）
    # size表示可视化的行列数（e.g. [10×10]）
    m = images.shape[0]  # 图片总数
    for i in range(m):
        image = images[i, :].reshape([pixel[0], pixel[1]])
        plt.subplot(size[0], size[1], i+1)
        plt.imshow(image.T)
        plt.axis('off')
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
        #                     wspace=0, hspace=0)
    plt.show()


# %% Logistic回归
# 输入：特征x(m×(1+n)))，标签y(m×1)，迭代次数epochs，学习率alpha，正则化参数lamda
# 输出：参数theta((1+n)×1)
# ** Note: 输入矩阵x必须满足：(1)已经归一化; (2)第一列(截距项)为1
#          输出参数theta的第一个数为theta0（截距项系数）
def logistic(x, y, epochs=1000, alpha=0.01, lamda=0):
    m = x.shape[0]  # 样本数
    n = x.shape[1]  # 特征数
    theta = np.random.uniform(-1, 1, [x.shape[1], 1])  # 参数初始化
    delta = np.zeros([n, 1])  # 梯度初始化
    cost_history = {'epoch': [], 'cost': []}  # 字典记录误差变化
    # 训练
    # print('************ 开始训练 ************')
    for epoch in range(epochs):
        # 假设函数h(θ)
        h = sigmoid(np.matmul(x, theta))
        # 交叉熵损失 + 正则化项
        J = cross_entropy(h, y) + lamda * 1/(2*m) * np.sum(pow(theta[1:n, :], 2))
        # 计算梯度
        delta[0, :] = 1/m * np.matmul(x.T[0, :], h-y)  # theta0不加正则化
        delta[1:n, :] = 1/m * np.matmul(x.T[1:n, :], h-y) + lamda*1/m*theta[1:n, :]
        # 参数更新
        theta = theta - alpha * delta
        # 记录误差cost
        cost_history['epoch'].append(epoch)
        cost_history['cost'].append(J)
        # print('Epoch:%d, Cost:%.4f' % (epoch, J))
    # print('************ 训练完成 ************')
    # 可视化误差曲线
    plt.plot(cost_history['epoch'], cost_history['cost'])
    plt.show()

    return theta


# %% 一维特征扩展为高维多项式特征
def polyfeatures(x, degree):
    res = np.ones(x.shape)
    for i in range(1, degree+1):
        res = np.column_stack((res, pow(x, i)))
    return res


# %% 二维特征扩展为高维多项式特征
def feature_mapping(x1, x2, degree):
    # degree = 8
    res = np.ones([x1.shape[0], 1])
    for i in range(1, degree+1):  # i:1~degree
        for j in range(i+1):      # j:0~i
            new_feature = np.multiply(x1**(i-j), x2**j)
            res = np.column_stack((res, new_feature))
    return res


# %% 绘制原始样本分布
# 适用于变量数为2，类别数为2的情况
# 输入np数组data（m×3）：共三列。第一、二列为x1、x2，第三列为分类标签label（取值：0和1）
def plot_original_data(data):
    colors = ['c', 'orange']
    marker = ['o', 's']
    for i in range(2):
        x1 = data[data[:, 2] == i][:, 0]
        x2 = data[data[:, 2] == i][:, 1]
        plt.scatter(x1, x2, c=colors[i], marker=marker[i], s=50, linewidths=0.8, edgecolors='k')


# %% 绘制决策边界
# 适用于变量数为2，类别数为2的情况
# 输入np数组data（m×3）：共三列。第一、二列为x1、x2，第三列为分类标签label（取值：0和1）
# 输入线性模型theta（3×1）: 注意theta包含三列（含截距项），分别为：θ0，θ1，θ2
def plot_decision_boundary(data, theta):
    # 函数功能：根据x和θ求出预测值，并以0.5为阈值将样本分类
    def predict(x, theta):
        predx = np.matmul(x, theta)
        return np.array([1 if predx[i] > 0.5 else 0 for i in range(x.shape[0])])
    # 生成二维网格
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 得到判别结果
    xygrid = feature_mapping(xx.ravel(), yy.ravel(), degree=1)  # 加上第一列（全1）
    Z = predict(xygrid, theta)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  # 绘制决策边界
    plot_original_data(data)  # 绘制原始样本分布
    plt.show()


# %% K-means算法
# 输入：data矩阵（m×n）：m个样本，n个特征；K：想要聚为多少类；iteration：迭代次数
# 输出：c（m×1）为所有样本的聚类结果（范围：0~K-1），centroid（K×n）：K个聚类中心的坐标
def kmeans(data, K, iteration=1000):
    # 求a, b的欧式距离
    def distance(a, b):
        return pow((sum(pow(a - b, 2))), 0.5)

    # 随机选择初始聚类中心
    m = data.shape[0]  # 样本数m
    index = np.random.random_integers(1, m, K)  # 在1~m中产生K个随机数
    centroid = data[index]  # 初始聚类中心
    d = np.zeros((m, K))  # 距离矩阵
    c = np.zeros(m)  # 类别向量(0,1,...,K)
    for iter in range(iteration):
        print('Iteration:', iter)
        # 簇分配
        c_last = c  # 保存上一次迭代的分类结果
        for i in range(m):
            for k in range(K):
                # 计算第i个样本距离第k个聚类中心的距离
                d[i, k] = distance(data[i], centroid[k])
        c = d.argmin(axis=1)  # 按行取最小值，得到每个样本所属类别
        # 终止条件：没有样本被重新分类
        if (c == c_last).all():
            break
        # 移动聚类中心
        for k in range(K):
            # 找到所有属于第k类的样本
            examples_belong_to_k = data[[i for i, x in enumerate(c) if x == k]]
            centroid[k] = np.mean(examples_belong_to_k, 0)

    return c, centroid


# %% PCA算法
# 输入：data是m×n矩阵，m为样本数，n为特征数（注意：data需要预先归一化！！！）
# 输出：u为m×m矩阵，用来对行降维（取u的前k列，得到m×k的u_reduce，然后用u_reduce.T乘x即可得到x的低维表示：k×m）
#      u为n×n矩阵，用来对列降维（取v的前k行，得到k×n的v_reduce，然后用x乘v_reduce.T即可得到x的低维表示：m×k）
#      k为最低的特征数目，满足“99% of variance is retained”
def pca(data):
    # 选择合适的K
    def find_k(s):
        # s表示对角阵，但是s是1×n矩阵，只列出了对角线元素
        k = s.shape[0]
        sum_s = np.sum(s)
        sum_k = sum_s
        while (sum_k / sum_s) >= 0.99:
            sum_k = sum_k - s[k - 1]
            k = k - 1
        return k + 1

    # sigma = np.matmul(data.T, data) / data.shape[0]  # 协方差矩阵
    # 上面一步求协方差和下面的svd函数中求协方差重了！（详见PCA_faces.py调试记录）
    [u, s, v] = np.linalg.svd(data)  # 调用svd函数
    # 注：返回的s并不是一个n×n矩阵，而是1×n的元组，表示对角线元素！
    k = find_k(s)  # 满足“99%的方差均被保留”的最小的k
    return u, v, k


# %% 异常值处理，提供三种方法
def Outlier(dataframe, method):
    for i in range(dataframe.shape[1]):
        data = dataframe.iloc[:, i]
        # 异常值检测
        if method == '3sigma':   # 3σ原则
            yc_bool_index = (data.mean()-3*data.std() > data) | \
                            (data.mean()+3*data.std() < data)

        if method == 'plotbox':  # 箱线图法
            QL = data.quantile(0.25)
            QU = data.quantile(0.75)
            IQR = QU - QL
            yc_bool_index = (data > QU + 1.5 * IQR) | (data < QL - 1.5 * IQR)

        if method == 'zscore':   # zscore法
            MAD = (data - data.median()).abs().median()
            # z_score = (data - data.mean()) / data.std()  # 普通zscore
            z_score = ((data - data.median()) * 0.6475 / MAD).abs()  # 增强zscore
            yc_bool_index = z_score.abs() > 3.5

        yc_index = np.arange(data.shape[0])[yc_bool_index]

        # 异常值处理
        data.iloc[yc_index] = data.mean()
        dataframe.iloc[:, i] = data

    return dataframe


# %% 定义DataSet类：包含了next_batch函数
class DataSet(object):

    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0  # 完成遍历轮数
        self._index_in_epochs = 0  # 调用next_batch()函数后记住上一次位置
        self._num_examples = images.shape[0]  # 训练样本数

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self._index_in_epochs
        # 初始化：将输入进行洗牌（只在最开始执行一次）
        if self._epochs_completed == 0 and start == 0 and shuffle:
            index0 = np.arange(self._num_examples)
            # print(index0)
            np.random.shuffle(index0)
            # print(index0)
            self._images = np.array(self._images)[index0]
            self._labels = np.array(self._labels)[index0]
            # print(self._images)
            # print(self._labels)
            # print("-----------------")

        # *特殊情况：取到最后，剩余样本数不足一个batch_size
        if start + batch_size > self._num_examples:
            # 先把剩余样本的取完
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # 重新洗牌，得到新版的数据集
            if shuffle:
                index = np.arange(self._num_examples)
                np.random.shuffle(index)
                self._images = self._images[index]
                self._labels = self._labels[index]
            # 再从新的数据集中取，补全batch_size个样本
            start = 0
            self._index_in_epochs = batch_size - rest_num_examples
            end = self._index_in_epochs
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            # 将旧的和新的拼在一起，得到特殊情况下的batch样本
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)

        # 正常情况：往后取batch_size个样本
        else:
            self._index_in_epochs += batch_size
            end = self._index_in_epochs
            return self._images[start:end], self._labels[start:end]


