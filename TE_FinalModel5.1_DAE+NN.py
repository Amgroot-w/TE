# ********************* Final Model -- 5.DAE降维+NN分类 **********************
"""
1. ” 特征提取 + 分类器 “（共保存三个模型）
（1）model1: 正常样本特征提取模型
（2）model2: 故障样本特征提取模型
（3）model3: 分类器模型

2. 特点: （1）三个模型可分别调节参数，互不影响;
        （2）通过第三个模型“分类器”的结果来调参;
        （3）关于数据集集的分配：
            model1的训练：正常样本（500个）单独训练；
            model2的训练：故障样本（480个）单独训练；
            model3的训练：前500个正常样本放入model1降维，后480个故障数据放入model2降维，然后再拼接得
                    到（980，20）训练集，将这个训练集放入model3训练
            model3的测试：前160个正常样本放入model1降维，后800个故障数据放入model2降维，然后再拼接得
                    到（960，20）测试集，将这个测试集放入model3测试

调试记录：
1. 第15类故障，测试的时候发现，在160之前，41~45段，136~140段也检测为故障。检查原始数据excel表发现：
（1）这两段数据的V36变量有较长的字符型数据，且最后一个V52变量为缺失值。这两个的处理方法分别是是：字符型数据取
前13个字符，已经改成了正常的数据范围；缺失值用前面的值填充，也是正常范围，说明问题不再这两项上；
（2）再次查看原始数据excel表发现，不仅上述两个变量有问题，还有好多个变量的范围都与其他时间段的变量范围差别较大，
据此推测可能是这个原因导致了这两个时间段被判定成了故障。


"""
# %% 导入包
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, manifold
import TE_function as TE
import csv

# %% 准备数据集
fault_classes = 21,  # 输入故障类别
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
NormalModel = TE.DAE(input_num=52, hidden_num=20, output_num=52,
                     learning_rate=0.05, dropout_keep_prob=0.8, epochs=1500,
                     lamda=0.01, batch_size=128)
# 模型1保存路径
model_path1 = "D:\\Python Codes\\TE\\TE_SaveDAE\\Feature/SaveDAE_Normal.ckpt"
# 绘制计算图
NormalModel.draw_ComputeMap()
# 训练
NormalModel.train(train_normal_x, model_path1)
# 得到特征
train_normal_feature = NormalModel.get_feature(train_normal_x)
test_normal_feature = NormalModel.get_feature(test_normal_x)


# %% 定义故障样本特征提取模型
FaultModel = TE.DAE(input_num=52, hidden_num=20, output_num=52,
                    learning_rate=0.05, dropout_keep_prob=0.8, epochs=1500,
                    lamda=0.01, batch_size=128)
# 模型2保存路径
model_path2 = "D:\\Python Codes\\TE\\TE_SaveDAE\\Fault/SaveDAE_Fault.ckpt"
# 绘制计算图
FaultModel.draw_ComputeMap()
# 训练
FaultModel.train(train_fault_x, model_path2)
# 得到特征
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
# 模型3保存路径
model_path3 = "D:\\Python Codes\\TE\\TE_SaveDAE\\Classification/SaveDAE_NeuralNetwork.ckpt"
# 训练
ClassifyModel.train(train_feature, train_y, model_path3)
# 测试
ClassifyModel.test(test_feature, test_y)


# %% 测试集的故障实时监控图
test_pred = ClassifyModel.predict(test_feature)
test_fault_prob = test_pred[:, 1]
TE.test_plot(test_fault_prob)

