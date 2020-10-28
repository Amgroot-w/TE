# ********************* Logistic分类器 **********************
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import TE_function as TE
import cap

fault_classes = 6,  # 输入故障类别
_, train_y = TE.get_data('train', fault_classes)
_, test_y = TE.get_data('test', fault_classes)

# 读取提取到的特征
train_x = pd.read_csv('train_feature.csv', header=None).values
test_x = pd.read_csv('test_feature.csv', header=None).values

logi_x = np.column_stack((np.ones(train_x.shape[0]), train_x))
logi_y = train_y
logi_y = np.array([0 if logi_y[i, 0] == 1 else 1 for i in range(logi_y.shape[0])]).reshape([-1, 1])
theta = cap.logistic(logi_x, logi_y, epochs=5000, alpha=1, lamda=10)  # 逻辑回归
logi_pred0 = cap.sigmoid(np.matmul(logi_x, theta))
logi_pred = np.array([0 if logi_pred0[i] < 0.5 else 1 for i in range(logi_pred0.shape[0])]).reshape([-1, 1])
ac = np.mean((logi_pred == 1) & (logi_y == 1)) + np.mean((logi_pred == 0) & (logi_y == 0))
print('训练集准确率:', '%.2f' % (ac * 100), '%')

logi_x = np.column_stack((np.ones(test_x.shape[0]), test_x))
logi_y = test_y
logi_y = np.array([0 if logi_y[i, 0] == 1 else 1 for i in range(logi_y.shape[0])]).reshape([-1, 1])
logi_pred0 = cap.sigmoid(np.matmul(logi_x, theta))
logi_pred = np.array([0 if logi_pred0[i] < 0.5 else 1 for i in range(logi_pred0.shape[0])]).reshape([-1, 1])
ac = np.mean((logi_pred == 1) & (logi_y == 1)) + np.mean((logi_pred == 0) & (logi_y == 0))
print('测试集准确率:', '%.2f' % (ac * 100), '%')