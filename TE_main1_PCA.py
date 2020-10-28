# ********************* PCA特征提取 **********************

import numpy as np
import csv
import TE_function as TE
import cap

fault_classes = 15,  # 输入故障类别
train_x, train_y = TE.get_data('train', fault_classes)
test_x, test_y = TE.get_data('test', fault_classes)

# 归一化
train_x = cap.normalize(train_x)
test_x = cap.normalize(test_x)
# PCA降维
u, v, k = cap.pca(train_x)
# 得到低维表示
k = 20

train_feature = np.matmul(train_x, v[:k, :].T)
test_feature = np.matmul(test_x, v[:k, :].T)

# 将特征写入csv文件
TE.SaveFeature(train_feature, test_feature)

