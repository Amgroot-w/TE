# ********************* 神经网络分类器 **********************
# 自己手写实现

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import TE_function as TE

import cap

fault_classes = 1,  # 输入故障类别（其实这个数不用改，因为是2分类，onehot编码之后都一样）
_, train_y = TE.get_data('train', fault_classes)
_, test_y = TE.get_data('test', fault_classes)

# 读取提取到的特征
train_x = pd.read_csv('train_feature.csv', header=None).values
test_x = pd.read_csv('test_feature.csv', header=None).values
train_num = train_x.shape[0]

# 设置超参数
input_num = train_x.shape[1]  # 输入节点数
hidden_num = 8  # 隐层节点数
output_num = train_y.shape[1]  # 输出节点数
alpha = 0.5  # 学习率
lamda = 0  # 正则化参数
epochs = 2000  # 迭代次数

# 初始化权重
w1 = np.random.uniform(-0.5, 0.5, [input_num, hidden_num])  # 普通初始化方法
w2 = np.random.uniform(-0.5, 0.5, [hidden_num, output_num])
# threshold = np.sqrt(6) / np.sqrt(400+25)
# w1 = np.random.uniform(-threshold, threshold, [input_num, hidden_num])  # 公式初始化方法
# threshold = np.sqrt(6) / np.sqrt(25+10)
# w2 = np.random.uniform(-threshold, threshold, [hidden_num, output_num])
b1 = np.zeros(hidden_num)
b2 = np.zeros(output_num)

# 训练
cost = []
for epoch in range(epochs):
    # 前向传播
    hidden_in = np.matmul(train_x, w1) + b1
    hidden_out = cap.sigmoid(hidden_in)
    network_in = np.matmul(hidden_out, w2) + b2
    network_out = cap.softmax(network_in)

    # 记录总误差
    J = cap.cross_entropy(network_out, train_y) \
        + 1/(2*train_num) * lamda * (np.sum(w2**2) + np.sum(w1**2))
    cost.append(J)
    # 反向传播
    # output_delta = np.multiply(network_out - train_y,
    #                            np.multiply(network_out, 1-network_out))
    output_delta = network_out - train_y

    hidden_delta = np.multiply(np.matmul(output_delta, w2.T),
                               np.multiply(hidden_out, 1-hidden_out))
    # 梯度更新
    dw2 = 1/train_num * (np.matmul(hidden_out.T, output_delta) + lamda*w2)
    db2 = 1/train_num * np.matmul(np.ones([train_num, 1]).T, output_delta)
    dw1 = 1/train_num * (np.matmul(train_x.T, hidden_delta) + lamda*w1)
    db1 = 1/train_num * np.matmul(np.ones([train_num, 1]).T, hidden_delta)
    w2 = w2 - alpha*dw2
    w1 = w1 - alpha*dw1
    b2 = b2 - alpha*db2
    b1 = b1 - alpha*db1
    # 展示训练过程
    if epoch % 200 == 0:
        print('Epoch:%4d     cost:%.4f' % (epoch, J))
# 可视化cost曲线
plt.plot(range(epochs), cost)
plt.show()

# 评估模型分类效果
accuracy = np.mean(np.equal(np.argmax(network_out, 1), np.argmax(train_y, 1)))
print("训练集测试准确率：%.2f%s" % (accuracy*100, '%'))
hidden_in = np.matmul(test_x, w1) + b1
hidden_out = cap.sigmoid(hidden_in)
network_in = np.matmul(hidden_out, w2) + b2
network_out = cap.sigmoid(network_in)
accuracy = np.mean(np.equal(np.argmax(network_out, 1), np.argmax(test_y, 1)))
print("测试集测试准确率：%.2f%s" % (accuracy*100, '%'))

