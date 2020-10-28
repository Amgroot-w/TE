# **************** 循环神经网络RNN -- 多个故障分类 *****************
# RNN类型：n vs.n
"""
调试记录：
1. 发现规律：时间深度T越大，学习率越大，都会使cost出现nan的情况提前，所以通过
减小时间深度T和学习率alpha，可以延缓cost变成nan的时间。

2. 未解决：对于11类故障的分类，准确率太低，模型训练失败。

"""

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import TE_function as TE
import cap


#%% 建立RNN数据集（一般为3维矩阵）
fault_classes = 1, 2, 4, 6, 7, 11, 13, 14, 17, 19, 20  # 选择故障类型
train_x0, train_y0 = TE.get_data('train', fault_classes)
test_x0, test_y0 = TE.get_data('test', fault_classes)

# # 特征归一化
# normalize = preprocessing.MinMaxScaler()
# train_x0 = normalize.fit_transform(train_x0)
# test_x0 = normalize.transform(test_x0)

T = 4  # RNN时间深度（必须满足能被样本总数整除）
train_x = train_x0.reshape(-1, T, train_x0.shape[1])  # 三维训练样本
train_y = train_y0.reshape(-1, T, train_y0.shape[1])  # 三维训练标签
m = train_x.shape[0]  # RNN训练样本数


#%% 配置网络参数
input_num = 52  # 输入节点数
hidden_num = 20  # 隐层节点数
output_num = train_y[0].shape[1]  # 输出节点数

alpha = 0.00001  # 学习率
epochs = 1  # 迭代次数


#%% 参数初始化
u = np.random.uniform(-0.5, 0.5, [input_num, hidden_num])
v = np.random.uniform(-0.5, 0.5, [hidden_num, output_num])
w = np.random.uniform(-0.5, 0.5, [hidden_num, hidden_num])
b = np.zeros([1, hidden_num])
c = np.zeros([1, output_num])

du = np.zeros_like(u)
dw = np.zeros_like(w)
dv = np.zeros_like(v)
db = np.zeros_like(b)
dc = np.zeros_like(c)


#%% 训练
cost = []
for epoch in range(epochs):
    for i in range(m):
        error = 0  # 误差累积值初始化
        hidden_values = list()  # 隐层状态初始化
        output_delta = np.zeros([T, train_y0.shape[1]])  # 输出层误差初始化
        # 随时间前向传播
        for t in range(T):
            # 读取数据：第i个训练样本、第t个时间的输入
            input_x = train_x[i, t, :].reshape(1, -1)
            input_y = train_y[i, t, :].reshape(1, -1)
            # 第t个RNN模型中，空间上前向传播
            if t == 0:
                hidden_in = np.matmul(input_x, u) + b
            else:
                hidden_in = np.matmul(input_x, u) + np.matmul(hidden_values[-1], w) + b
            hidden_out = cap.sigmoid(hidden_in)
            network_in = np.matmul(hidden_out, v) + c
            network_out = cap.softmax(network_in)  # 多分类，softmax输出
            # 记录隐层状态
            hidden_values.append(hidden_out)
            # 记录损失
            pred = network_out
            error += cap.cross_entropy(pred, input_y)
            # 记录误差delta
            output_delta[t, :] = pred - input_y

        # cost.append(error)  # 记录误差

        # 随时间反向传播（BPTT）
        for t in reversed(range(T)):
            # 读取数据：第i个训练样本、第t个时间的输入
            input_x = train_x[i, t, :].reshape(1, -1)
            input_y = train_y[i, t, :].reshape(1, -1)
            # 误差反向传播
            # 中间隐层是线性输出，公式里面含有sigmoid的导数项
            if t == T-1:
                hidden_delta = np.multiply(np.matmul(output_delta[t], v.T),
                                           np.multiply(hidden_values[t], 1-hidden_values[t]))
            else:
                hidden_delta = np.multiply(np.matmul(delta_future, w),
                                           np.multiply(hidden_values[t], 1-hidden_values[t])) \
                               + np.multiply(np.matmul(output_delta[t], v.T),
                                             np.multiply(hidden_values[t], 1-hidden_values[t]))
            delta_future = hidden_delta

            # 求梯度
            du += np.matmul(input_x.T, hidden_delta)
            dv += np.matmul(hidden_values[t].T, output_delta[t].reshape(1, -1))
            dw += np.matmul(hidden_values[t-1].T, hidden_delta)
            db += hidden_delta
            dc += output_delta[t]

        # 更新
        u -= alpha * du
        v -= alpha * dv
        w -= alpha * dw
        b -= alpha * db
        c -= alpha * dc

        # 梯度值重置
        du *= 0
        dv *= 0
        dw *= 0
        db *= 0
        dc *= 0

    cost.append(error)  # 记录误差
    print('Epoch:%d   Cost:%.4f' % (epoch, error))

plt.plot(range(len(cost)), cost)
plt.show()


#%% 测试
def predict(x0):
    x = x0.reshape(-1, T, x0.shape[1])
    res = []
    for i_ in range(x.shape[0]):
        hidden_values_ = list()
        for t_ in range(T):
            input_x_ = x[i_, t_, :].reshape(1, -1)
            if t == 0:
                hidden_in_ = np.matmul(input_x_, u) + b
            else:
                hidden_in_ = np.matmul(input_x_, u) + np.matmul(hidden_values_[-1], w) + b
            hidden_out_ = cap.sigmoid(hidden_in_)
            network_in_ = np.matmul(hidden_out_, v) + c
            network_out_ = cap.softmax(network_in_)

            hidden_values.append(hidden_out_)
            res.append(network_out_)
    return np.array(res)


train_pred = predict(train_x0).reshape(train_y0.shape)
test_pred = predict(test_x0).reshape(test_y0.shape)
train_pred = np.argmax(train_pred, axis=1).reshape(-1, 1)
test_pred = np.argmax(test_pred, axis=1).reshape(-1, 1)
train_true = np.argmax(train_y0, axis=1).reshape(-1, 1)
test_true = np.argmax(test_y0, axis=1).reshape(-1, 1)

print('训练集准确率：%.2f' % (100*np.mean(train_pred == train_true)), '%')
print('测试集准确率：%.2f' % (100*np.mean(test_pred == test_true)), '%')










