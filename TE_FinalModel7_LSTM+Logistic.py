# ********************* Final Model -- 7.LSTM&DAE降维+Logistic分类 **********************
"""
调试记录：

1. 因为(980,)和(980,1)这个bug，又浪费了快1个小时才找出来。。。
随手reshape(-1,1)是个好习惯！！！（或者keepdims=True）；

2. 调试成功后，发现logistic和NN的效果差不多；

"""
# %% 导入包
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing, manifold
import TE_function as TE
import csv


# %% 准备数据集
fault_classes = 3,  # 输入故障类别
train_x, train_y = TE.get_data('train', fault_classes)
test_x, test_y = TE.get_data('test', fault_classes)
normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)
# 分配数据集
train_normal_x = train_x[:500, :]
train_fault_x = train_x[500:980, :]
test_normal_x = test_x[:160, :]
test_fault_x = test_x[160:960, :]


# %% 定义正常样本特征提取模型
# 注意，LSTM的训练结构要求：batch_size必须能够被time_step整除
NormalModel = TE.LSTM_DAE(input_num=52, hidden_num=20, output_num=52,
                          time_step=10, batch_size=100, learning_rate=0.05,
                          lamda=0.001, dropout_keep_prob=0.8, epochs=1000)
# 创建一张新的计算图
g1 = tf.Graph()
with g1.as_default():
    # 绘制计算图
    NormalModel.draw_ComputeMap()
    # 模型保存路径
    model_path1 = "D:\\Python Codes\\TE\\TE_SaveLSTM\\Feature/SaveLSTM_Normal.ckpt"
    # 训练
    NormalModel.train(train_normal_x, model_path1)
    # 获取特征
    train_normal_feature = NormalModel.get_feature(train_normal_x)
    test_normal_feature = NormalModel.get_feature(test_normal_x)


# %% 定义故障样本特征提取模型
FaultModel = TE.LSTM_DAE(input_num=52, hidden_num=20, output_num=52,
                         time_step=10, batch_size=100, learning_rate=0.05,
                         lamda=0.001, dropout_keep_prob=0.8, epochs=1000)
# 创建一张新的计算图
g2 = tf.Graph()
with g2.as_default():
    # 绘制计算图
    FaultModel.draw_ComputeMap()
    # 模型保存路径
    model_path2 = "D:\\Python Codes\\TE\\TE_SaveLSTM\\Fault/SaveLSTM_Fault.ckpt"
    # 训练
    FaultModel.train(train_fault_x, model_path2)
    # 获取特征
    train_fault_feature = FaultModel.get_feature(train_fault_x)
    test_fault_feature = FaultModel.get_feature(test_fault_x)


# %% 整合提取到的特征（正常特征、故障特征），并2D可视化
train_feature = np.row_stack((train_normal_feature, train_fault_feature))
test_feature = np.row_stack((test_normal_feature, test_fault_feature))
# 将训练集随机打乱（测试集保持原来的顺序，不打乱）
train_data = np.column_stack((train_feature, train_y))
np.random.shuffle(train_data)
train_feature = train_data[:, :20]
train_y = train_data[:, 20:22]
# t-SNE二维可视化高维特征
print('\n训练集t-SNE拟合... ...')
train_tsne = manifold.TSNE(n_components=2).fit_transform(train_feature)
TE.visualize_feature(fault_classes[0], train_tsne, train_y)
print('测试集t-SNE拟合... ...\n')
test_tsne = manifold.TSNE(n_components=2).fit_transform(test_feature)
TE.visualize_feature(fault_classes[0], test_tsne, test_y)


# %% 训练Logistic分类器
train_y = np.argmax(train_y, axis=1).reshape([-1, 1])
test_y = np.argmax(test_y, axis=1).reshape([-1, 1])
# 定义逻辑回归模型
model = TE.Logi(learning_rate=1.0, lamda=0, epochs=30000)
# 训练
model.train(train_feature, train_y)
# 测试
model.test(train_feature, train_y, '训练集')
model.test(test_feature, test_y, '测试集')


# %% 测试集的故障实时监控图
test_fault_prob = model.predict(test_feature)
TE.test_plot(test_fault_prob)

