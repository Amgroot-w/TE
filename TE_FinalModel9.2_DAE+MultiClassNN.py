# ********************* Final Model -- 9.LSTM&DAE分别提取特征再堆叠+NN多分类 **********************
"""
调试记录：

"""
# %% 导入包
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing, manifold
import TE_function as TE

# %% 准备数据集：大数据集，含1类正常5类故障
fault_classes = 1, 2, 4, 6, 7, 18  # 输入故障类别
train_x, train_y = TE.get_data('train', fault_classes)
fault_classes = 0, 1, 2, 5, 6, 7, 18  # 输入故障类别
test_x, test_y = TE.get_data('test', fault_classes)
normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)


# %% 特征提取模型
model = TE.DAE(input_num=52, hidden_num=20, output_num=52,
               learning_rate=0.05, dropout_keep_prob=0.8, epochs=1500,
               lamda=0.03, batch_size=256)
# 绘制计算图
model.draw_ComputeMap()
# 模型保存路径
model_path1 = "D:\\Python Codes\\TE\\TE_SaveMulti\\FeatureDAE/SaveFeatureDAE.ckpt"
# 训练
model.train(train_x, model_path1)
# 获取特征
train_feature = model.get_feature(train_x)
test_feature = model.get_feature(test_x)

# t-SNE二维可视化高维特征
def visualize_feature(samples, labels, title_):
    data = np.column_stack((samples, labels))
    label = ['Normal', 'Fault1', 'Fault6', 'Fault9', 'Fault14', 'Fault18', ]  # 标注
    colors = ['gray', 'orange', 'cyan', 'yellowgreen', 'red', 'magenta']  # 点的颜色
    marker = ['o', 'o', 'o', 'o', 'o', 'o']  # 点的形状
    s = [30, 30, 30, 30, 30, 30]  # 点的大小
    edgecolors = ['k', 'k', 'k', 'k', 'k', 'k']  # 边缘线颜色
    linewidths = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # 边缘线宽度
    alphas = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
    for i in range(6):
        x1 = data[data[:, 2] == i][:, 0]
        x2 = data[data[:, 2] == i][:, 1]
        plt.scatter(x1, x2, c=colors[i], marker=marker[i], label=label[i],
                    s=s[i], linewidths=linewidths[i], edgecolors=edgecolors[i],
                    alpha=alphas[i])
    plt.title(title_)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


# train_label = np.argmax(train_y, axis=1).reshape([-1, 1])
# test_label = np.argmax(test_y, axis=1).reshape([-1, 1])
# print('\n训练集t-SNE拟合... ...')
# train_tsne = manifold.TSNE(n_components=2).fit_transform(train_feature)
# title = "TrainSet: Feature Visualization of All 5 Faults"
# visualize_feature(train_tsne, train_label, title)
# print('测试集t-SNE拟合... ...\n')
# test_tsne = manifold.TSNE(n_components=2).fit_transform(test_feature)
# title = "TestSet: Feature Visualization of All 5 Faults"
# visualize_feature(test_tsne, test_label, title)


# %% 训练分类器
ClassifyModel = TE.NeuralNetwork(input_num=20, hidden_num=40, output_num=len(fault_classes),
                                 learning_rate=0.01, epochs=1000, decay_steps=100, decay_rate=0.8,
                                 lamda=0.001, batch_size=256, MultiClass=True)
# 模型保存路径
model_path2 = "D:\\Python Codes\\TE\\TE_SaveMulti\\Classification/SaveNN.ckpt"
# 训练
ClassifyModel.train(train_feature, train_y, model_path2)
# 测试
ClassifyModel.test(test_feature, test_y)


# %% 测试集的故障实时监控图
test_pred = ClassifyModel.predict(test_feature)
title_name = ['Fault1', 'Fault6', 'Fault9', 'Fault14', 'Fault18', ]
count_sum = 0
for k in np.arange(1, len(fault_classes)):
    start = 960 + 960*(k-1)
    end = start + 960
    test_fault_prob = test_pred[start:end, k].reshape([-1, 1])
    TE.test_plot(test_fault_prob)







