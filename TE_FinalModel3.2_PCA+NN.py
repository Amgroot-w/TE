# ********************* Final Model -- 3.PCA降维+NN分类 **********************
"""
调试记录：

"""
# %% 导入包
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing, manifold
import TE_function as TE
import cap


# %% 准备数据集
fault_classes = 18,  # 输入故障类别
train_x, train_y = TE.get_data('train', fault_classes)
test_x, test_y = TE.get_data('test', fault_classes)
normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)


# %% 定义特征提取模型
FaultModel = TE.PCA(pc_num=20)  # 实例化一个PCA模型
FaultModel.fit(train_x)  # 故障样本
# 得到特征
train_feature = FaultModel.get_feature(train_x)
test_feature = FaultModel.get_feature(test_x)

# 将训练集随机打乱（测试集保持原来的顺序，不打乱）
train_data = np.column_stack((train_feature, train_y))
np.random.shuffle(train_data)
train_feature = train_data[:, :20]
train_y = train_data[:, 20:22]

# t-SNE二维可视化高维特征
print('\n训练集t-SNE拟合... ...')
train_tsne = manifold.TSNE(n_components=2).fit_transform(train_feature)
title = 'TrainSet: Feature Visulization of Fault ' + str(fault_classes[0])
TE.visualize_feature(train_tsne, train_y, title)
print('测试集t-SNE拟合... ...\n')
test_tsne = manifold.TSNE(n_components=2).fit_transform(test_feature)
title = 'TestSet: Feature Visulization of Fault ' + str(fault_classes[0])
TE.visualize_feature(test_tsne, test_y, title)


# %% 训练分类器
ClassifyModel = TE.NeuralNetwork(input_num=20, hidden_num=10, output_num=2,
                                 learning_rate=0.1, epochs=300, decay_steps=100, decay_rate=0.6,
                                 lamda=0, batch_size=256)
# 模型保存路径
model_path3 = "D:\\Python Codes\\TE\\TE_SaveLSTM\\Classification/SaveLSTM_NeuralNetwork.ckpt"
# 训练
ClassifyModel.train(train_feature, train_y, model_path3)
# 测试
ClassifyModel.test(test_feature, test_y)


# %% 测试集的故障实时监控图
test_pred = ClassifyModel.predict(test_feature)
test_fault_prob = test_pred[:, 1]
TE.test_plot(test_fault_prob)






