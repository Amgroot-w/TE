# ********************* SVM分类器 **********************
# 调用sklearn的包

import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import TE_function as TE

fault_classes = 6,  # 输入故障类别
_, train_y = TE.get_data('train', fault_classes, onehot=None)
_, test_y = TE.get_data('test', fault_classes, onehot=None)

# 读取提取到的特征
train_x = pd.read_csv('train_feature.csv', header=None).values
test_x = pd.read_csv('test_feature.csv', header=None).values


# # 网格搜索：找到最优的参数C、gamma组合
# para1 = np.arange(4, 5, 0.1)
# para2 = np.arange(4, 5, 0.1)
# test_score = np.zeros([len(para1), len(para2)])
# best_index = [0, 0]
# for i in range(len(para1)):
#     for j in range(len(para2)):
#         clf = svm.SVC(kernel='rbf', C=para1[i], gamma=para2[j])
#         clf.fit(train_x, train_y)
#         train_ac = clf.score(train_x, train_y)
#         test_ac = clf.score(test_x, test_y)
#         test_score[i, j] = test_ac
#         if test_ac > test_score[best_index[0], best_index[1]]:
#             test_index = [i, j]
#         print(i, '/', len(para1)-1, '\t\t', j, '/', len(para2)-1)
# xx = range(len(para1))
# yy = range(len(para2))
# xx, yy = np.meshgrid(xx, yy)
# plt.contour(xx, yy, test_score)
# plt.show()

# 用最优的参数C、gamma来分类
# C = para1[best_index[0]]
# gamma = para2[best_index[1]]
clf = svm.SVC(kernel='rbf')
clf.fit(train_x, train_y)
train_ac = clf.score(train_x, train_y)
test_ac = clf.score(test_x, test_y)
print("训练集测试准确率：%.2f%s" % (train_ac*100, '%'))
print("测试集测试准确率：%.2f%s" % (test_ac*100, '%'))



