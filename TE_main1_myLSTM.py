# **************** LSTM 特征提取  *****************
# RNN类型：n vs.n
"""
调试记录：
1. 时间深度T=2和T=20，训练出来的自编码器特诊提取后，在用分类器分类，
效果差不了多少，准确率几乎一样。。。
2. 迭代次数为10：测试集准确率60%左右；
   迭代次数为100，测试集准确率97%左右。
"""

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import TE_function as TE
import cap

np.random.seed(5)

#%% 建立LSTM数据集（一般为3维矩阵）
fault_classes = 6,  # 选择故障类型
train_x0, _ = TE.get_data('train', fault_classes)
test_x0, _ = TE.get_data('test', fault_classes)

normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x0 = normalize.fit_transform(train_x0)
test_x0 = normalize.transform(test_x0)

T = 20  # LSTM时间深度（必须满足能被样本总数980、960整除）
train_x = train_x0.reshape(-1, T, train_x0.shape[1])  # 三维训练样本
test_x = test_x0.reshape(-1, T, test_x0.shape[1])  # 三维测试样本
m = train_x.shape[0]  # LSTM训练样本数


# %% 配置网络参数
input_dim = 52   # 输入维度
hidden_dim = 20  # 隐层输出维度
cell_dim = 20    # 细胞状态维度
output_dim = 52  # 输出维度

alpha = 0.001  # 学习率
lamda = 0.1  # 正则化参数
epochs = 100  # 迭代次数


# %% 参数初始化
sigma = 0.5
w_fh = np.random.uniform(-sigma, sigma, [cell_dim, hidden_dim])
w_ih = np.random.uniform(-sigma, sigma, [cell_dim, hidden_dim])
w_ch = np.random.uniform(-sigma, sigma, [cell_dim, hidden_dim])
w_oh = np.random.uniform(-sigma, sigma, [cell_dim, hidden_dim])

w_fx = np.random.uniform(-sigma, sigma, [cell_dim, input_dim])
w_ix = np.random.uniform(-sigma, sigma, [cell_dim, input_dim])
w_cx = np.random.uniform(-sigma, sigma, [cell_dim, input_dim])
w_ox = np.random.uniform(-sigma, sigma, [cell_dim, input_dim])

b_f = np.zeros([1, cell_dim])
b_i = np.zeros([1, cell_dim])
b_c = np.zeros([1, cell_dim])
b_o = np.zeros([1, cell_dim])

v = np.random.uniform(-sigma, sigma, [hidden_dim, output_dim])
c = np.zeros([1, output_dim])

# %% 训练
cost = []
for epoch in range(epochs):
    for k in range(m):
        # 初始化
        # 储存每个时刻的加权输入
        net = {
            'f': np.zeros([T, cell_dim]),
            'i': np.zeros([T, cell_dim]),
            'c': np.zeros([T, cell_dim]),
            'o': np.zeros([T, cell_dim])
        }
        # 储存每个时刻的门、候选细胞状态
        door = {
            'f': np.zeros([T, cell_dim]),
            'i': np.zeros([T, cell_dim]),
            'c': np.zeros([T, cell_dim]),
            'o': np.zeros([T, cell_dim])
        }
        # 储存每个时刻的隐层状态、细胞状态
        state = {
            'hidden': np.zeros([T, hidden_dim]),
            'cell': np.zeros([T, cell_dim])
        }
        # 初始化各参数梯度
        d_wfh = np.zeros_like(w_fh)
        d_wfx = np.zeros_like(w_fx)
        d_wih = np.zeros_like(w_ih)
        d_wix = np.zeros_like(w_ix)
        d_wch = np.zeros_like(w_ch)
        d_wcx = np.zeros_like(w_cx)
        d_woh = np.zeros_like(w_oh)
        d_wox = np.zeros_like(w_ox)
        d_bf = np.zeros_like(b_f)
        d_bi = np.zeros_like(b_i)
        d_bc = np.zeros_like(b_c)
        d_bo = np.zeros_like(b_o)

        # 初始化误差error
        error = 0

        # 初始化输出层误差
        output_delta = []

        # forward propagation
        for t in range(T):
            # 得到t时刻的输入x
            input_x = train_x[k, t, :].reshape(1, -1)
            # 读取第i次训练的输出y
            input_y = train_x[k, t, :].reshape(1, -1)
            # t-1时刻的隐层状态、t-1时刻的细胞状态
            if t == 0:
                cell_pre = np.zeros([1, cell_dim])
                hidden_pre = np.zeros([1, hidden_dim])
            else:
                cell_pre = state['cell'][t-1:t, :]
                hidden_pre = state['hidden'][t-1:t, :]

            # 空间上前向传播：隐层
            # t时刻的：加权输入
            net_f = np.matmul(hidden_pre, w_fh.T) + np.matmul(input_x, w_fx.T) + b_f
            net_i = np.matmul(hidden_pre, w_ih.T) + np.matmul(input_x, w_ix.T) + b_i
            net_c = np.matmul(hidden_pre, w_ch.T) + np.matmul(input_x, w_cx.T) + b_c
            net_o = np.matmul(hidden_pre, w_oh.T) + np.matmul(input_x, w_ox.T) + b_o

            # t时刻的：门、候选细胞状态
            f = cap.sigmoid(net_f)
            i = cap.sigmoid(net_i)
            cell = cap.tanh(net_c)
            o = cap.sigmoid(net_o)

            # t时刻的：隐层状态、细胞状态
            cell_now = f * cell_pre + i * cell
            hidden_now = o * cap.tanh(cell_now)

            # 保存中间变量
            net['f'][t, :] = net_f
            net['i'][t, :] = net_i
            net['c'][t, :] = net_c
            net['o'][t, :] = net_o

            door['f'][t, :] = f
            door['i'][t, :] = i
            door['c'][t, :] = cell
            door['o'][t, :] = o

            state['cell'][t, :] = cell_now
            state['hidden'][t, :] = hidden_now

            # 空间上前向传播：输出层
            network_in = np.matmul(hidden_now, v) + c
            network_out = cap.sigmoid(network_in)

            pred = network_out

            # 均方误差损失 + 正则化项
            error += 1/2 * np.sum((pred - input_y)**2) + 1/2 * 1/train_x0.shape[0] * lamda \
                     * (np.sum(w_fh**2) + np.sum(w_fx**2) + np.sum(w_ih**2) + np.sum(w_ix**2)
                        + np.sum(w_ch**2) + np.sum(w_cx**2) + np.sum(w_oh**2) + np.sum(w_ox**2))

            output_delta.append((pred-input_y) * network_out * (1-network_out))  # 输出层误差delta

        # 随时间前向传播完成
        cost.append(error)  # 记录本次迭代的cost
        output_delta = np.array(output_delta).reshape(T, output_dim)  # 转化为np数组

        # back prapagation through time
        for t in reversed(range(T)):
            # 得到t时刻的输入x、
            input_x = train_x[k, t, :].reshape(1, -1)

            # 计算隐层误差 spcae
            hidden_delta_space = np.matmul(output_delta[t:t+1, :], v.T)

            # 计算隐层误差 time
            # 最后一层没有有时间上反向传播的误差
            if t == T - 1:
                hidden_delta_time = np.zeros([1, cell_dim])
            else:
                # 根据上一层的4个小误差，算本层的BPTT误差
                hidden_delta_time = np.matmul(delta_f, w_fh) + np.matmul(delta_i, w_ih) + \
                                    np.matmul(delta_c, w_ch) + np.matmul(delta_o, w_oh)

            # 计算总误差（时间+空间）
            hidden_delta = hidden_delta_space + hidden_delta_time

            # 计算4个小误差
            if t == 0:
                delta_f = np.zeros([1, cell_dim])
            else:
                delta_f = hidden_delta * door['o'][t, :] * (1 - cap.tanh(door['c'][t, :]) ** 2) \
                          * state['cell'][t - 1, :] * door['f'][t, :] * (1 - door['f'][t, :])

            delta_i = hidden_delta * door['o'][t, :] * (1 - cap.tanh(door['c'][t, :]) ** 2) \
                      * door['c'][t, :] * door['f'][t, :] * (1 - door['f'][t, :])

            delta_c = hidden_delta * door['o'][t, :] * (1 - cap.tanh(door['c'][t, :]) ** 2) \
                      * door['i'][t, :] * (1 - door['c'][t, :] ** 2)

            delta_o = hidden_delta * cap.tanh(door['c'][t, :]) \
                      * door['o'][t, :] * (1 - door['o'][t, :])

            # 计算梯度
            if t == 0:
                d_wfh += np.matmul(delta_f.T, np.zeros([1, cell_dim]))
                d_wih += np.matmul(delta_i.T, np.zeros([1, cell_dim]))
                d_wch += np.matmul(delta_c.T, np.zeros([1, cell_dim]))
                d_woh += np.matmul(delta_o.T, np.zeros([1, cell_dim]))
            else:
                d_wfh += np.matmul(delta_f.T, state['hidden'][t-1:t, :]) + lamda * w_fh
                d_wih += np.matmul(delta_i.T, state['hidden'][t-1:t, :]) + lamda * w_ih
                d_wch += np.matmul(delta_c.T, state['hidden'][t-1:t, :]) + lamda * w_ch
                d_woh += np.matmul(delta_o.T, state['hidden'][t-1:t, :]) + lamda * w_oh

            d_wfx += np.matmul(delta_f.T, input_x) + lamda * w_fx
            d_wix += np.matmul(delta_i.T, input_x) + lamda * w_ix
            d_wcx += np.matmul(delta_c.T, input_x) + lamda * w_cx
            d_wox += np.matmul(delta_o.T, input_x) + lamda * w_ox

            d_bf += delta_f + lamda * b_f
            d_bi += delta_i + lamda * b_i
            d_bc += delta_c + lamda * b_c
            d_bo += delta_o + lamda * b_o

        # 参数更新
        w_fh -= alpha * d_wfh
        w_ih -= alpha * d_wih
        w_ch -= alpha * d_wch
        w_oh -= alpha * d_woh

        w_fx -= alpha * d_wfx
        w_ix -= alpha * d_wix
        w_cx -= alpha * d_wcx
        w_ox -= alpha * d_wox

        b_f -= alpha * d_bf
        b_i -= alpha * d_bi
        b_c -= alpha * d_bc
        b_o -= alpha * d_bo

    print('Epoch:%3d  Cost:%.4f' % (epoch, error))

plt.plot(range(len(cost)), cost)
plt.show()


# %% 测试
def predict(x):
    _feature = []
    _pred = []
    for _k in range(x.shape[0]):
        # 储存每个时刻的隐层状态、细胞状态
        _state = {
            'hidden': np.zeros([T, hidden_dim]),
            'cell': np.zeros([T, cell_dim])
        }
        # forward propagation
        for _t in range(T):
            # 得到t时刻的输入x
            _input_x = x[_k, _t, :].reshape(1, -1)
            # t-1时刻的隐层状态、t-1时刻的细胞状态
            if _t == 0:
                _cell_pre = np.zeros([1, cell_dim])
                _hidden_pre = np.zeros([1, hidden_dim])
            else:
                _cell_pre = _state['cell'][_t-1:_t, :]
                _hidden_pre = _state['hidden'][_t-1:_t, :]

            # t时刻的：门、候选细胞状态
            _f = cap.sigmoid(np.matmul(_hidden_pre, w_fh.T) + np.matmul(_input_x, w_fx.T) + b_f)
            _i = cap.sigmoid(np.matmul(_hidden_pre, w_ih.T) + np.matmul(_input_x, w_ix.T) + b_i)
            _cell = cap.tanh(np.matmul(_hidden_pre, w_ch.T) + np.matmul(_input_x, w_cx.T) + b_c)
            _o = cap.sigmoid(np.matmul(_hidden_pre, w_oh.T) + np.matmul(_input_x, w_ox.T) + b_o)

            # t时刻的：隐层状态、细胞状态
            _cell_now = _f * _cell_pre + _i * _cell
            _hidden_now = _o * cap.tanh(_cell_now)
            _feature.append(_hidden_now)

            # 空间上前向传播：输出层
            _network_out = cap.sigmoid(np.matmul(_hidden_now, v) + c)
            _pred.append(_network_out)

    _feature = np.array(_feature).reshape(x.shape[0]*x.shape[1], hidden_dim)
    _pred = np.array(_pred).reshape(x.shape[0]*x.shape[1], x.shape[2])

    return _pred, _feature


train_pred, train_feature = predict(train_x)
test_pred, test_feature = predict(test_x)
train_cost = np.mean(np.sum(pow(train_pred - train_x0, 2), axis=1))
test_cost = np.mean(np.sum(pow(test_pred - test_x0, 2), axis=1))

print("训练集的重建误差为：%.4f" % train_cost)
print("测试集的重建误差为：%.4f" % test_cost)

# %% 保存特征
TE.SaveFeature(train_feature, test_feature)








