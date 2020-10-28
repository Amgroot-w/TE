# **************** RNN 特征提取  *****************
# RNN类型：n vs.n
"""
调试记录：
1. 误差反向传播的公式写错了：漏写了最后一项sigmoid的导数，导致重建误差一直在10左右，
尽管增大epoch，误差也不会降低，反而会逐渐升高。改正公式后，可以看到cost曲线明显的下降
趋势，最终的重建误差从10左右降低个位数，且增大epoch，会继续降低，表明训练有了效果。但
是问题是不管epoch增大到多少，最终的分类器的准确率并没有提升。
总结：epoch增大的话，增加了计算量，但是准确率并没有明显提高，所以epoch设置为1.

"""

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import TE_function as TE
import cap

np.random.seed(0)


#%% 建立RNN数据集（一般为3维矩阵）
fault_classes = 6,  # 选择故障类型
train_x0, _ = TE.get_data('train', fault_classes)
test_x0, _ = TE.get_data('test', fault_classes)

# 特征归一化
normalize = preprocessing.MinMaxScaler()
train_x0 = normalize.fit_transform(train_x0)
test_x0 = normalize.transform(test_x0)

T = 4  # RNN时间深度（必须满足能被样本总数980、960整除）
train_x = train_x0.reshape(-1, T, train_x0.shape[1])  # 三维训练样本
m = train_x.shape[0]  # RNN训练样本数


#%% 配置网络参数
input_num = 52  # 输入节点数
hidden_num = 20  # 隐层节点数
output_num = 52  # 输出节点数

alpha = 0.0005  # 学习率
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
        hidden_values = []  # 隐层状态初始化
        output_delta = np.zeros([T, output_num])  # 输出层误差初始化
        # 随时间前向传播
        for t in range(T):
            # 读取数据：第i个训练样本、第t个时间的输入
            input_x = train_x[i, t, :].reshape(1, -1)
            input_y = train_x[i, t, :].reshape(1, -1)
            # 第t个RNN模型中，空间上前向传播
            if t == 0:
                hidden_in = np.matmul(input_x, u) + b
            else:
                hidden_in = np.matmul(input_x, u) + np.matmul(hidden_values[-1], w) + b
            hidden_out = hidden_in  # 提取特征：线性输出
            network_in = np.matmul(hidden_out, v) + c
            network_out = cap.sigmoid(network_in)  # 重建输入：sigmoid输出
            # 记录隐层状态
            hidden_values.append(hidden_out)
            # 记录损失
            pred = network_out
            # error += np.sum((pred - input_y)**2)  # 均方误差损失
            error += cap.mse(pred, input_y)  # capcapcapcapcapcap
            # 记录误差delta
            output_delta[t] = np.multiply(pred - input_y, np.multiply(network_out, 1-network_out))

        cost.append(error)  # 记录误差

        # 随时间反向传播（BPTT）
        for t in reversed(range(T)):
            # 读取数据：第i个训练样本、第t个时间的输入
            input_x = train_x[i, t, :].reshape(1, -1)
            input_y = train_x[i, t, :].reshape(1, -1)
            # 误差反向传播
            # 注意：中间隐层是线性输出，因此下面的公式不含sigmoid的导数项
            if t == T-1:
                hidden_delta = np.matmul(output_delta[t], v.T).reshape(1, -1)
            else:
                hidden_delta = (np.matmul(delta_future, w) + np.matmul(output_delta[t], v.T)).reshape(1, -1)
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

    # cost.append(error)  # 记录误差
    print('Epoch:%d   Cost:%.4f' % (epoch, error))

plt.plot(range(len(cost)), cost)
plt.show()


#%% 测试
def predict(x0):
    x = x0.reshape(-1, T, x0.shape[1])
    _pred = []
    _feature = []
    for i_ in range(x.shape[0]):
        hidden_values_ = list()
        for t_ in range(T):
            input_x_ = x[i_, t_, :].reshape(1, -1)
            if t == 0:
                hidden_in_ = np.matmul(input_x_, u) + b
            else:
                hidden_in_ = np.matmul(input_x_, u) + np.matmul(hidden_values_[-1], w) + b
            hidden_out_ = hidden_in_
            network_in_ = np.matmul(hidden_out_, v) + c
            network_out_ = cap.sigmoid(network_in_)

            hidden_values_.append(hidden_out_)

            _feature.append(hidden_out_)  # 保存特征值
            _pred.append(network_out_)  # 保存重建值

    _pred = np.array(_pred).reshape(x0.shape)
    _feature = np.array(_feature).reshape(x0.shape[0], hidden_num)

    return _pred, _feature


train_pred, train_feature = predict(train_x0)
test_pred, test_feature = predict(test_x0)
train_cost = np.mean(np.sum(pow(train_pred - train_x0, 2), axis=1))
test_cost = np.mean(np.sum(pow(test_pred - test_x0, 2), axis=1))

print("训练集的重建误差为：%.4f" % train_cost)
print("测试集的重建误差为：%.4f" % test_cost)

#%% 保存特征
TE.SaveFeature(train_feature, test_feature)





