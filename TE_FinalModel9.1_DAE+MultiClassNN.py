# ********************* Final Model -- 9.LSTM&DAE分别提取特征再堆叠+NN多分类 **********************
"""
调试记录：
1. 发现准确率、F1-Sscore、FAR、MDR的计算方式好像有错（不管怎样都会输出1、100%）

2. 多故障类别分别提取特征，在堆叠在一起分类，效果和单独分类一样，效果都超好

"""
# %% 导入包
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing, manifold
import TE_function as TE

# 准备数据集：大数据集，含1类正常5类故障
fault_classes = 0, 1, 8, 13, 15, 21  # 输入故障类别
train_x, train_y = TE.get_data('train', fault_classes)
train_x = train_x[500:3400, :]
train_y = train_y[500:3400, :]
test_x, test_y = TE.get_data('test', fault_classes)
normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)


# %% 6个特征提取模型
graph = list()
train_data = np.zeros([1, 26])
test_data = np.zeros([1, 26])
for i in range(6):
    train_xx = train_x[train_y[:, i] == 1]
    train_yy = train_y[train_y[:, i] == 1]
    test_xx = test_x[test_y[:, i] == 1]
    test_yy = test_y[test_y[:, i] == 1]

    model = TE.DAE(input_num=52, hidden_num=20, output_num=52,
                   learning_rate=0.05, dropout_keep_prob=0.8, epochs=1500,
                   lamda=0.01, batch_size=128)
    # 创建一张新的计算图
    g = tf.Graph()
    graph.append(g)
    with graph[i].as_default():
        # 绘制计算图
        model.draw_ComputeMap()
        # 模型保存路径
        model_path1 = "D:\\Python Codes\\TE\\TE_SaveMulti\\FeatureDAE/SaveFeatureDAE.ckpt"
        # 训练
        model.train(train_xx, model_path1)
        # 获取特征
        train_ff = model.get_feature(train_xx)
        test_ff = model.get_feature(test_xx)

        train_data = np.row_stack((train_data, np.column_stack((train_ff, train_yy))))
        test_data = np.row_stack((test_data, np.column_stack((test_ff, test_yy))))

train_data = train_data[1:2901, :]
test_data = test_data[1:5761, :]


# %% 整理数据集
# np.random.shuffle(train_data)  # 打乱训练集
train_feature = train_data[:, :20]
test_feature = test_data[:, :20]
train_label = np.argmax(train_data[:, 20:26], axis=1).reshape([-1, 1])
test_label = np.argmax(test_data[:, 20:26], axis=1).reshape([-1, 1])

# t-SNE二维可视化高维特征
def visualize_feature(samples, labels, title_):
    data = np.column_stack((samples, labels))
    label = ['Normal', 'Fault1', 'Fault8', 'Fault13', 'Fault15', 'Fault21', ]  # 标注
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


print('\n训练集t-SNE拟合... ...')
train_tsne = manifold.TSNE(n_components=2).fit_transform(train_feature)
title = "TrainSet: Feature Visualization of All 5 Faults"
visualize_feature(train_tsne, train_label, title)
print('测试集t-SNE拟合... ...\n')
test_tsne = manifold.TSNE(n_components=2).fit_transform(test_feature)
title = "TestSet: Feature Visualization of All 5 Faults"
visualize_feature(test_tsne, test_label, title)


# %% 训练分类器
ClassifyModel = TE.NeuralNetwork(input_num=20, hidden_num=40, output_num=6,
                                 learning_rate=0.01, epochs=1000, decay_steps=100, decay_rate=0.8,
                                 lamda=0.001, batch_size=256, MultiClass=True)
# 模型保存路径
model_path2 = "D:\\Python Codes\\TE\\TE_SaveMulti\\Classification/SaveNN.ckpt"
# 训练
ClassifyModel.train(train_feature, train_data[:, 20:26], model_path2)
# 测试
ClassifyModel.test(test_feature, test_data[:, 20:26])


# %% 测试集的故障实时监控图
test_pred = ClassifyModel.predict(test_feature)
title_name = ['Fault1', 'Fault8', 'Fault13', 'Fault15', 'Fault21', ]
count_sum = 0
for k in np.arange(1, 6):
    start1 = 960 + 160*(k-1)
    end1 = start1 + 160
    test_normal_prob = test_pred[start1:end1, k].reshape([-1, 1])

    start2 = 1760 + 800*(k-1)
    end2 = start2 + 800
    test_fault_prob = test_pred[start2:end2, k].reshape([-1, 1])
    alltime_prob = np.row_stack((test_normal_prob, test_fault_prob))

    # plt.subplot(1, 6, k)
    plt.plot(np.arange(0, 960), 0.5 * np.ones(960), 'k--', label='Threshold: 50%')
    plt.plot(np.arange(0, 160), alltime_prob[:160], 'b', label='Feature State')
    plt.plot(np.arange(160, 960), alltime_prob[160:960], 'r', label='Fault State')
    plt.title(title_name[k-1])
    plt.xlabel('Test samples')
    plt.ylabel('Fault Probability')
    plt.legend()
    plt.show()

    # 测试集故障检出点、漏检个数
    detected_point = 0
    for i in range(960):
        if alltime_prob[i] > 0.5:
            break
        else:
            detected_point += 1
    count = 0
    for j in np.arange(160, 960):
        if alltime_prob[j] <= 0.5:
            count += 1
    count_sum += count
    print('第%d个测试集：故障检出点：%d，漏检个数：%d\n' % (k, detected_point, count))
print('总漏检个数：%d' % count_sum)





