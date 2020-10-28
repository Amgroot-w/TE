# ********** Final MultiClass Model -- 6.LSTM降维 + NN多分类 *****************
"""
调试记录：

"""
# %% 导入包
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
import TE_function as TE
import time
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
start_time = time.time()


# %% 准备数据集
# fault_names = ['Normal', 'Fault1', 'Fault2', 'Fault6', 'Fault14', 'Fault18']
fault_names = ['正常', '故障1', '故障2', '故障6', '故障14', '故障18']
fault_classes = 1, 2, 6, 14, 18
train_x, train_y = TE.get_data('train', fault_classes)
fault_classes = 0, 1, 2, 6, 14, 18
test_x, test_y = TE.get_data('test', fault_classes)
normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)


# %% 特征提取模型
model = TE.LSTM_DAE(input_num=52, hidden_num=24, output_num=52,
                    time_step=4, batch_size=256, learning_rate=0.012,
                    lamda=0, dropout_keep_prob=1, epochs=1000)
# 绘制计算图
model.draw_ComputeMap()
# 模型保存路径
model_path1 = "D:\\Python Codes\\TE\\TE_SaveMulti\\FeatureDAE/SaveFeatureDAE.ckpt"
# 训练
model.train(train_x, model_path1)
# 获取特征
train_feature = model.get_feature(train_x)
test_feature = model.get_feature(test_x)
# # t-SNE二维可视化高维特征
# TE.tSNE_visualize(fault_names, train_feature, train_y, test_feature, test_y, title_name='LSTM')


# %% 训练分类器
ClassifyModel = TE.NeuralNetwork(input_num=24, hidden_num=40, output_num=len(fault_classes),
                                 learning_rate=0.01, epochs=1000, decay_steps=100, decay_rate=0.8,
                                 lamda=0, batch_size=256, MultiClass=True)
# 模型保存路径
model_path2 = "D:\\Python Codes\\TE\\TE_SaveMulti\\Classification/SaveNN.ckpt"
# 训练
ClassifyModel.train(train_feature, train_y, model_path2)
# 测试
ClassifyModel.test(test_feature, test_y)


# %% 测试集的故障实时监控图
test_pred = ClassifyModel.predict(test_feature)
count_sum = 0
for k in range(len(fault_classes)):
    start = 960 * k
    end = start + 960
    if k == 0:
        test_normal_prob = np.argmax(test_pred[start:end, :], axis=1).reshape([-1, 1])
        count = 960 - np.sum(test_normal_prob == 0)
        print('%s：误报个数：%d，误报率：%.2f%s' % (fault_names[k], count, (100 * count/960), '%'))
    else:
        test_fault_prob = test_pred[start:end, k].reshape([-1, 1])
        TE.test_plot(fault_names[k], test_fault_prob)

end_time = time.time()
print('运行时间：%.3f s' % (end_time - start_time))





