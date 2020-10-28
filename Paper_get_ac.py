
# %% 导入包
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import TE_function as TE
import csv

# %% 准备数据集
fault_names = ['Normal', 'Fault1', 'Fault2', 'Fault6', 'Fault14', 'Fault18']
fault_classes = 1, 2, 6, 14, 18
train_x, train_y = TE.get_data('train', fault_classes)
fault_classes = 0, 1, 2, 6, 14, 18
test_x, test_y = TE.get_data('test', fault_classes)
normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)

para1s = [2, 4, 5, 10, 20]  # 时间深度
batch_sizes = [256, 256, 200, 200, 200]
para2s = [16, 18, 20, 22, 24]  # 隐层节点数
train_ac = np.zeros([len(para1s), len(para2s)])
test_ac = np.zeros([len(para1s), len(para2s)])
graph = list()
k = 0
for i, para1 in enumerate(para1s):
    for j, para2 in enumerate(para2s):
        # %% 特征提取模型
        model = TE.LSTM_DAE(input_num=52, hidden_num=para2, output_num=52,
                            time_step=para1, batch_size=batch_sizes[i], learning_rate=0.012,
                            lamda=0, dropout_keep_prob=1, epochs=500)
        # 绘制计算图
        g = tf.Graph()
        graph.append(g)
        with graph[k].as_default():
            model.draw_ComputeMap()
            # 模型保存路径
            model_path1 = "D:\\Python Codes\\TE\\TE_SaveMulti\\FeatureDAE/SaveFeatureDAE.ckpt"
            # 训练
            model.train(train_x, model_path1)
            # 获取特征
            train_feature = model.get_feature(train_x)
            test_feature = model.get_feature(test_x)

            # %% 训练分类器
            ClassifyModel = TE.NeuralNetwork(input_num=para2, hidden_num=40, output_num=len(fault_classes),
                                             learning_rate=0.01, epochs=500, decay_steps=100, decay_rate=0.8,
                                             lamda=0, batch_size=batch_sizes[i], MultiClass=True)
            # 模型保存路径
            model_path2 = "D:\\Python Codes\\TE\\TE_SaveMulti\\Classification/SaveNN.ckpt"
            # 训练
            ClassifyModel.train(train_feature, train_y, model_path2)
            train_ac[i][j] = ClassifyModel.ac
            # 测试
            ClassifyModel.test(test_feature, test_y)
            test_ac[i][j] = ClassifyModel.ac
            k += 1

with open('train_ac.csv', 'w') as datafile:
    writer = csv.writer(datafile, delimiter=',')
    writer.writerows(train_ac)

with open('test_ac.csv', 'w') as datafile:
    writer = csv.writer(datafile, delimiter=',')
    writer.writerows(test_ac)



