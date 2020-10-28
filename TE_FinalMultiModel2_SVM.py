# ********** Final MultiClass Model -- 2.SVM直接多分类 *****************
"""
调试记录:

"""
# %% 导入包
import numpy as np
from sklearn import preprocessing, svm
import matplotlib.pyplot as plt
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
train_x, train_y = TE.get_data('train', fault_classes, onehot=False)
fault_classes = 0, 1, 2, 6, 14, 18
test_x, test_y = TE.get_data('test', fault_classes, onehot=False)
# 特征归一化
normalize = preprocessing.MinMaxScaler()
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)

# %% SVM拟合
model = svm.SVC(kernel='rbf', gamma=24.4, probability=True)  # 定义一个svm模型
model.fit(train_x, train_y)  # 模型拟合
train_accuracy = model.score(train_x, train_y)  # 训练集准确率
test_accuracy = model.score(test_x, test_y)  # 测试集准确率
train_pred = model.predict_proba(train_x)  # 训练集预测值
test_pred = model.predict_proba(test_x)  # 测试集预测值

# %% 评价指标
print("训练集测试准确率：%.2f%s" % (100 * train_accuracy, '%'))
print("测试集测试准确率：%.2f%s" % (100 * test_accuracy, '%'))

# %% 测试集的故障实时监控图
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


