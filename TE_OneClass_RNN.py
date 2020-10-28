# **************** 循环神经网络RNN -- 单个故障分类 *****************
# RNN类型：n vs.n
"""
调试记录：
1. epoch设置为1，根本就没有学到东西！看着准确率还挺高（训练集48.98%，测试集83.33%），但
是这其实是因为模型不管输入是什么，一律输出1，而本来标签为1的样本就很多，所以准确率误以为很高，
其实模型根本没有训练出效果；把epoch调高之后，就会出现cost曲线反复跳变，虽然整体趋势有时候
会下降，但是中间过程震荡的太剧烈了。
2. 鉴于以上问题，采用F1-score，代替ac来评价模型分类效果。
3. 学习率设置特别低:alpha=0.00001，迭代200次，得到一个不错的模型，查看pred的值，发现并没
有出现大量无脑预测1的情况，说明RNN训练的比较好了！***只有一点：cost曲线太丑了（剧烈震荡）
4. 上述3的调试结果是第6类故障的，一旦换了故障（例如换乘第5类），整个评价指标就立即变低了，还
得重新调参，才能达到与3中相同的效果。
5. 误差反向传播的公式写错了：漏写了sigmoid的导数项，加上之后重新调参，效果和之前一样比较好。
6. 未解决问题：为什么学习率必须特别小？？？？？？(梯度爆炸/消失 ？？？)
稍微大一点点，cost就提前出现nan了（例如alpha等于0.01时，cost从第十几代开始就变成nan了）
7. 时间深度T增大为20，发现准确率和T=4是差不多一样的，可见增加深度并没有提升效果。
"""

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import TE_function as TE
import cap

np.random.seed(0)


#%% 建立RNN数据集（一般为3维矩阵）
fault_classes = 6,  # 选择故障类型
train_x0, train_y0 = TE.get_data('train', fault_classes, normalize=False, onehot=False)
test_x0, test_y0 = TE.get_data('test', fault_classes, normalize=False, onehot=False)

# 特征归一化
normalize = preprocessing.MinMaxScaler()
train_x0 = normalize.fit_transform(train_x0)
test_x0 = normalize.transform(test_x0)

# 将标签的取值限定在0（正常）和1（故障）
train_y0 = np.array([0 if train_y0[i] == 0 else 1 for i in range(len(train_y0))]).reshape(-1, 1)
test_y0 = np.array([0 if test_y0[i] == 0 else 1 for i in range(len(test_y0))]).reshape(-1, 1)

T = 4  # RNN时间深度（必须满足能被样本总数980、960整除）
train_x = train_x0.reshape(-1, T, train_x0.shape[1])  # 三维训练样本
train_y = train_y0.reshape(-1, T, 1)  # 三维训练标签
m = train_x.shape[0]  # RNN训练样本数


#%% 配置网络参数
input_num = 52  # 输入节点数
hidden_num = 20  # 隐层节点数
output_num = 1  # 输出节点数

alpha = 0.00001  # 学习率
epochs = 100  # 迭代次数


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
tidu = []  # 记录梯度值，观察梯度消失、梯度爆炸现象
cost = []  # 记录误差变化
for epoch in range(epochs):
    for i in range(m):
        error = 0  # 误差累积值初始化
        hidden_values = list()  # 隐层状态初始化
        output_delta = np.zeros([T, output_num])  # 输出层误差初始化
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
            network_out = cap.sigmoid(network_in)  # 二分类，sigmoid输出
            # 记录隐层状态
            hidden_values.append(hidden_out)
            # 记录损失
            pred = network_out
            error += cap.cross_entropy(pred, input_y)
            # 记录误差delta
            output_delta[t] = pred - input_y

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
            dv += np.matmul(hidden_values[t].T, output_delta[t].reshape(-1, 1))
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
            if i == 0:
                tidu.append((np.mean(du**2) + np.mean(dv**2) +
                            np.mean(dw**2) + np.mean(db**2) + np.mean(dc**2))/5)
            du *= 0
            dv *= 0
            dw *= 0
            db *= 0
            dc *= 0

    cost.append(error)  # 记录误差
    print('Epoch:%d   Cost:%.4f' % (epoch, error))

plt.plot(range(len(cost)), cost)
plt.show()
plt.plot(range(len(tidu)), tidu)
plt.show()

#%% 测试
def predict(x0):
    x = x0.reshape(-1, T, x0.shape[1])
    res = []
    for i in range(x.shape[0]):
        hidden_values_ = list()
        for t_ in range(T):
            input_x_ = x[i, t_, :].reshape(1, -1)
            if t == 0:
                hidden_in_ = np.matmul(input_x_, u) + b
            else:
                hidden_in_ = np.matmul(input_x_, u) + np.matmul(hidden_values_[-1], w) + b
            hidden_out_ = cap.sigmoid(hidden_in_)
            network_in_ = np.matmul(hidden_out_, v) + c
            network_out_ = cap.sigmoid(network_in_)

            hidden_values.append(hidden_out_)
            res.append(network_out_)
    res = [1 if res[i] > 0.5 else 0 for i in range(len(res))]
    return np.array(res)


train_pred = predict(train_x0).reshape(-1, 1)
test_pred = predict(test_x0).reshape(-1, 1)

print('训练集准确率：%.2f' % (100*np.mean(train_pred == train_y0)), '%，',
      '训练集F1-Score：%.2f' % cap.F1_Score(train_pred, train_y0),
      '\t(训练集预测输出1占比：%.2f' % (100*np.mean(train_pred)), '%，',
      '真实占比：%.2f' % (100*np.mean(train_y0)), '%)')
print('测试集准确率：%.2f' % (100*np.mean(test_pred == test_y0)), '%，',
      '测试集F1-Score：%.2f' % cap.F1_Score(test_pred, test_y0),
      '\t(测试集预测输出1占比：%.2f' % (100*np.mean(test_pred)), '%，',
      '真实占比：%.2f' % (100*np.mean(test_y0)), '%)')










