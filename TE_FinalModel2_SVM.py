# ********************* Final Model -- 2.SVM直接分类 **********************
"""
调试记录:
1. svm对于简单的故障类别，效果挺好的，漏检个数在10以内，但是对于难的故障类别
仍然效果不好。

"""
# %% 导入包
import numpy as np
from sklearn import preprocessing, svm
import matplotlib.pyplot as plt
import TE_function as TE
import cap


# %% 准备数据集
fault_class = 1, 2, 4, 6, 18
train_x, train_y = TE.get_data('train', fault_class, onehot=False)
test_x, test_y = TE.get_data('test', fault_class, onehot=False)
# 特征归一化
normalize = preprocessing.MinMaxScaler()
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)


# %% SVM拟合
model = svm.SVC(kernel='rbf', gamma=24.4, probability=True)  # 定义一个svm模型
model.fit(train_x, train_y)  # 模型拟合
train_accuracy = model.score(train_x, train_y)  # 训练集准确率
test_accuracy = model.score(test_x, test_y)  # 测试集准确率
train_pred = model.predict(train_x)  # 训练集预测值
test_pred = model.predict(test_x)  # 测试集预测值


# %% 评价指标
print("训练集测试准确率：%.2f%s" % (100 * train_accuracy, '%'))
print("测试集测试准确率：%.2f%s" % (100 * test_accuracy, '%'))

print("训练集F1-Score：%.4f" % cap.F1_Score(train_pred, train_y))
print("测试集F1-Score：%.4f" % cap.F1_Score(test_pred, test_y))

train_FAR, train_MDR = cap.FAR_MDR(train_pred, train_y)
test_FAR, test_MDR = cap.FAR_MDR(test_pred, test_y)
print("训练集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s" % (100 * train_FAR, '%', 100 * train_MDR, '%'))
print("测试集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s" % (100 * test_FAR, '%', 100 * test_MDR, '%'))

# 测试集的故障实时监控图
test_fault_prob = model.predict_proba(test_x)[:, 1]
TE.test_plot(test_fault_prob)
