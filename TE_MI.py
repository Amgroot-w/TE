# 计算变量之间的互信息
"""
怎么计算故障之间的互信息啊？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cap

data = pd.read_csv('TE_Train_data.csv')  # 导入原始数据
data.iloc[:, :52] = cap.normalize(data.iloc[:, :52])  # 特征归一化

MI = []  # 储存每个特征的互信息
for j in range(52):
    a = np.log(np.var(data.iloc[:, j]))

    b = 0
    for i in range(22):
        data_x = data.loc[data['V52'] == i].iloc[:, j]  # 第i个类别的第j个特征
        b += data_x.shape[0] / data.shape[0] * np.log(np.var(data_x))

    MI.append(1/2 * (a - b))

plt.bar(range(len(MI)), MI)
plt.xlabel('Variables')
plt.ylabel('Mutual Information')
plt.show()












