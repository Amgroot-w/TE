# ********** Final MultiClass Model -- 5.DAE降维 + NN多分类 *****************
"""
调试记录：
1. 不能在一张计算图上定义两个basic_LSTM_cell！！！刚开始代码采用一张默认计算图来
训练两个故障提取模型，但是一直报错；后来将正常样本特征提取模型、故障样本特征提取模型
分别绘制在两张计算图上进行训练，便解决了这个问题。
2. 对于好分的&不好分的故障，SVM分类效果都能达到100%（故障3、故障15都是100% ！！！）
3. 发现准确率、F1-Sscore、FAR、MDR的计算方式好像有错（不管怎样都会输出1、100%）
4. 多故障类别分别提取特征，在堆叠在一起分类，效果和单独分类一样，效果都超好

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
model = TE.DAE(input_num=52, hidden_num=22, output_num=52,
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

# # t-SNE二维可视化高维特征
# TE.tSNE_visualize(fault_names, train_feature, train_y, test_feature, test_y, title_name='DAE')


# %% 训练分类器
ClassifyModel = TE.NeuralNetwork(input_num=22, hidden_num=34, output_num=len(fault_classes),
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





