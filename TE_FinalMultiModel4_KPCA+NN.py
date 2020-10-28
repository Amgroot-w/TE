# ********** Final MultiClass Model -- 4.KPCA降维 + NN多分类 *****************
"""
调试记录：

"""
# %% 导入包
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, manifold
from sklearn.decomposition import KernelPCA
import TE_function as TE
import cap
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


# %% 定义特征提取模型
kpca2 = KernelPCA(kernel='rbf', n_components=20)
kpca2.fit(train_x)
# 得到特征
train_feature = kpca2.transform(train_x)
test_feature = kpca2.transform(test_x)
# # t-SNE二维可视化高维特征
# TE.tSNE_visualize(fault_names, train_feature, train_y, test_feature, test_y, title_name='KPCA')


# %% 训练分类器
ClassifyModel = TE.NeuralNetwork(input_num=20, hidden_num=10, output_num=len(fault_classes),
                                 learning_rate=0.5, epochs=500, decay_steps=100, decay_rate=0.6,
                                 lamda=0, batch_size=256, MultiClass=True)
# 模型保存路径
model_path3 = "D:\\Python Codes\\TE\\TE_SaveLSTM\\Classification/SaveLSTM_NeuralNetwork.ckpt"
# 训练
ClassifyModel.train(train_feature, train_y, model_path3)
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



