# ********************* TE_function.py **********************
"""
** 已放入python安装目录的lib文件夹下，可以直接import导入此包

实现功能：
1.得到所需数据（包含：一些数据预处理 & 索引目标文件数据）
2.异常值处理（包含三个处理方法）
3.DataSet类：数据集的next_batch
4.保存提取到的特征
5.可视化二维特征
6.DAE类：降噪自编码器
7.LASM_DAE类：引入LSTM的自编码器
8.NeuralNetwork类：神经网络分类器
9.Logi类：logistic分类器
10.PCA类：主成分分析学习特征提取
11.绘制故障实时监控图、故障检出点、漏检个数
12.t-SNE二维可视化高维特征

"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing, manifold
import csv
import os
import cap
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 关闭警告


# t-SNE二维可视化高维特征
def tSNE_visualize(fault_names, train_feature, train_y, test_feature, test_y, title_name):
    def tSNE_plot0(samples, labels, title_):
        data = np.column_stack((samples, labels))
        label = fault_names
        colors = ['white', 'orange', 'cyan', 'yellowgreen', 'red', 'magenta']  # 点的颜色
        marker = ['s', 'o', 'o', 'o', 'o', 'o']  # 点的形状
        s = [30, 30, 30, 30, 30, 30]  # 点的大小
        edgecolors = ['k', 'k', 'k', 'k', 'k', 'k']  # 边缘线颜色
        linewidths = [0.7, 0.5, 0.5, 0.5, 0.5, 0.5]  # 边缘线宽度
        alphas = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
        for i in range(6):
            x1 = data[data[:, 2] == i][:, 0]
            x2 = data[data[:, 2] == i][:, 1]
            plt.scatter(x1, x2, c=colors[i], marker=marker[i], label=label[i],
                        s=s[i], linewidths=linewidths[i], edgecolors=edgecolors[i],
                        alpha=alphas[i])
        plt.title('特征可视化：'+title_, fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('特征1', fontsize=20)
        plt.ylabel('特征2', fontsize=20)
        plt.legend(fontsize=11)
        plt.show()

    def tSNE_plot(samples, labels, title_):
        data = np.column_stack((samples, labels))
        label = fault_names
        colors = ['white', 'orange', 'cyan', 'yellowgreen', 'red', 'magenta']  # 点的颜色
        for i in np.arange(1, 6):
            plt.figure()
            # 正常样本聚类
            x1 = data[data[:, 2] == 0][:, 0]
            x2 = data[data[:, 2] == 0][:, 1]
            plt.scatter(x1, x2, c='white', marker='s', label=label[0], s=30,
                        linewidths=0.6, edgecolors='k', alpha=0.65)
            # 故障样本聚类
            x1 = data[data[:, 2] == i][:, 0]
            x2 = data[data[:, 2] == i][:, 1]
            plt.scatter(x1, x2, c=colors[i], marker='o', label=label[i], s=30,
                        linewidths=0.6, edgecolors='k', alpha=0.65)

            plt.title(title_+'：可视化'+label[i], fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlabel('特征1', fontsize=20)
            plt.ylabel('特征2', fontsize=20)
            plt.legend(fontsize=15)
            plt.show()

    train_label = np.argmax(train_y, axis=1).reshape([-1, 1])
    test_label = np.argmax(test_y, axis=1).reshape([-1, 1])
    # print('\n训练集t-SNE拟合... ...')
    # train_tsne = manifold.TSNE(n_components=2).fit_transform(train_feature)
    # tSNE_plot(train_tsne, train_label, title_name)
    print('测试集t-SNE拟合... ...\n')
    test_tsne = manifold.TSNE(n_components=2).fit_transform(test_feature)
    tSNE_plot0(test_tsne, test_label, title_name)  # 全部故障聚类
    tSNE_plot(test_tsne, test_label, title_name)  # 单个故障聚类


# 绘制故障实时监控图、故障检出点、漏检个数
# 中文名称版
# 2021.3.10记录：画图竟然不保存，直接在sciview里面调取图片，真特么傻！
# （以后记住了：小论文所需的图往往需要多次修改，尽量保存一份！而且要尽量保存成svg格式、emf格式的，不要用png!）
def test_plot(fault_name, test_fault_prob):
    # 定义阈值
    threshold = 1/2 * (max(test_fault_prob) + min(test_fault_prob))
    # 测试集的故障实时监控图
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    # ax.plot(160 * np.ones(2), np.arange(0, 2), 'k--')
    ax.plot(np.arange(0, 960), threshold * np.ones(960), 'k--')
    ax.plot(np.arange(0, 160), test_fault_prob[:160], label='正常样本')
    ax.plot(np.arange(160, 960), test_fault_prob[160:960], label='故障样本')

    plt.text(880, threshold-0.18, '↑\n阈值: %.2f%s' % (100*threshold, '%'), ha='center', va='bottom', fontsize=17.5)
    ax.set_xticks([0, 80, 160, 240, 320, 400, 480, 560, 640, 720, 800, 880, 960])
    ax.set_xticklabels(['0', '80', '160\n↑\n引入故障', '240', '320', '400', '480', '560', '640', '720', '800', '880', '960'],
                       fontsize=17.5)
    plt.yticks(fontsize=17.5)

    ax.set_xlabel('测试样本序数', fontsize=20)
    ax.set_ylabel('故障概率', fontsize=20)
    ax.set_title('实时监测图: %s' % fault_name, fontsize=20)
    # ax.legend(fontsize=17.5)
    plt.show()

    # 测试集故障检出点
    for detected_point in np.arange(160, 960):
        if test_fault_prob[detected_point] > threshold:
            break
        else:
            detected_point += 1
    # 误报个数
    count1 = 0
    for j in np.arange(0, 160):
        if test_fault_prob[j] > threshold:
            count1 += 1
    # 误报率
    FAR = count1 / 160
    # 漏检个数
    count2 = 0
    for k in np.arange(160, 960):
        if test_fault_prob[k] <= threshold:
            count2 += 1
    # 漏检率
    MDR = count2 / 800
    print('%s：故障检出点：%d，误报个数：%d，漏检个数：%d, 误报率：%.2f%s，漏检率：%.2f%s'
          % (fault_name, detected_point+1, count1, count2, (100*FAR), '%', (100*MDR), '%'))

# 绘制故障实时监控图、故障检出点、漏检个数
# 英文名称版
def test_plot0(fault_name, test_fault_prob):
    # 定义阈值
    threshold = 1/2 * (max(test_fault_prob) + min(test_fault_prob))
    # 测试集的故障实时监控图
    plt.plot(np.arange(0, 960), threshold * np.ones(960), 'k--', label='Threshold: %.2f' % threshold)
    plt.plot(np.arange(0, 160), test_fault_prob[:160], 'b', label='Normal Probability')
    plt.plot(np.arange(160, 960), test_fault_prob[160:960], 'r', label='Fault Probability')
    plt.title('Real-time Monitoring: %s' % fault_name)
    plt.xlabel('Test samples')
    plt.ylabel('Fault Probability')
    plt.legend()
    plt.show()

    # 测试集故障检出点
    for detected_point in np.arange(160, 960):
        if test_fault_prob[detected_point] > threshold:
            break
        else:
            detected_point += 1
    # 误报个数
    count1 = 0
    for j in np.arange(0, 160):
        if test_fault_prob[j] > threshold:
            count1 += 1
    # 误报率
    FAR = count1 / 160
    # 漏检个数
    count2 = 0
    for k in np.arange(160, 960):
        if test_fault_prob[k] <= threshold:
            count2 += 1
    # 漏检率
    MDR = count2 / 800
    print('%s：故障检出点：%d，误报个数：%d，漏检个数：%d, 误报率：%.2f%s，漏检率：%.2f%s'
          % (fault_name, detected_point+1, count1, count2, (100*FAR), '%', (100*MDR), '%'))

# logistic分类器
class Logi(object):
    def __init__(self, learning_rate, lamda, epochs):
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.epochs = epochs

    def train(self, input_x, input_y):
        input_x = np.column_stack((np.ones([input_x.shape[0], 1]), input_x))
        input_y = input_y.reshape([-1, 1])
        m = input_x.shape[0]  # 样本数
        n = input_x.shape[1]  # 特征数
        self.theta = np.random.uniform(-1, 1, [n, 1])  # 参数初始化
        self.delta = np.zeros([n, 1])  # 梯度初始化
        self.cost_history = {'epoch': [], 'cost': []}  # 字典记录误差变化
        # 训练
        for epoch in range(self.epochs):
            # 假设函数h(θ)
            self.h = cap.sigmoid(np.matmul(input_x, self.theta))
            # 交叉熵损失 + 正则化项
            self.J = cap.cross_entropy(self.h, input_y) + self.lamda * 1/(2*m) * np.sum(pow(self.theta[1:n, :], 2))
            # 计算梯度
            self.delta[0, :] = 1/m * np.matmul(input_x.T[0, :], self.h - input_y)  # theta0不加正则化
            self.delta[1:n, :] = 1/m * np.matmul(input_x.T[1:n, :], self.h - input_y) \
                                 + self.lamda * 1/m * self.theta[1:n, :]
            # 参数更新
            self.theta = self.theta - self.learning_rate * self.delta
            # 记录误差cost
            self.cost_history['epoch'].append(epoch)
            self.cost_history['cost'].append(self.J)
        # 可视化误差曲线
        plt.plot(self.cost_history['epoch'], self.cost_history['cost'])
        plt.title('Logistic Classifiction')
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.show()

    def test(self, input_x, input_y, string):
        print(string)
        self.pred_prob = self.predict(input_x)
        self.pred = np.array([0 if self.pred_prob[i] < 0.5 else 1 for i in range(len(self.pred_prob))]).reshape([-1, 1])
        self.accuracy = np.mean(self.pred == input_y)
        # 训练集准确率
        print("\t准确率：%.2f%s" % (100*self.accuracy, '%'))
        # 训练集F1-Score
        print("\tF1-Score：%.4f" % cap.F1_Score(self.pred, input_y))
        # 训练集误报率FAR、漏检率MDR
        train_FAR, train_MDR = cap.FAR_MDR(self.pred, input_y)
        print("\t误报率FAR：%.2f%s，漏检率MDR：%.2f%s\n" % (100*train_FAR, '%', 100*train_MDR, '%'))

    def predict(self, input_x):
        input_xx = np.column_stack((np.ones([input_x.shape[0], 1]), input_x))
        return cap.sigmoid(np.matmul(input_xx, self.theta))


# %% PCA类
class PCA(object):
    def __init__(self, pc_num):
        self.pc_num = pc_num

    def fit(self, data):
        # sigma = np.matmul(data.T, data) / data.shape[0]  # 协方差矩阵
        # 上面一步求协方差和下面的svd函数中求协方差重了！（详见PCA_faces.py调试记录）
        self.u, self.s, self.v = np.linalg.svd(data)  # 调用svd函数
        # 注：返回的s并不是一个n×n矩阵，而是1×n的元组，表示对角线元素！
        # k = find_k(s)  # 满足“99%的方差均被保留”的最小的k

    def get_feature(self, data):
        feature = np.matmul(data, self.v[:self.pc_num, :].T)
        return feature

    @staticmethod
    def find_k(s):
        # 选择合适的K
        # s表示对角阵，但是s是1×n矩阵，只列出了对角线元素
        k = s.shape[0]
        sum_s = np.sum(s)
        sum_k = sum_s
        while (sum_k / sum_s) >= 0.99:
            sum_k = sum_k - s[k - 1]
            k = k - 1
        return k + 1


# DAE自编码器
class DAE(object):
    def __init__(self, input_num, hidden_num, output_num,
                 learning_rate, dropout_keep_prob, epochs,
                 lamda, batch_size):
        self.input_num = input_num    # 输入节点数
        self.hidden_num = hidden_num  # 隐层节点数
        self.output_num = output_num  # 输出节点数
        self.learning_rate = learning_rate  # 学习率
        self.dropout_keep_prob = dropout_keep_prob  # dropout参数
        self.epochs = epochs  # 迭代次数
        self.lamda = lamda    # 正则化参数
        self.batch_size = batch_size  # 批训练大小

    def draw_ComputeMap(self):
        # DAE网络输入
        self.x = tf.placeholder(tf.float32, [None, self.input_num])
        # 定义DAE网络权重
        self.weights = {
            'encoder_W': tf.Variable(tf.random_normal([self.input_num, self.hidden_num])),
            'decoder_W': tf.Variable(tf.random_normal([self.hidden_num, self.output_num]))
        }
        # 定义DAR网络偏置
        self.biases = {
            'encoder_b': tf.Variable(tf.zeros(self.hidden_num)),
            'decoder_b': tf.Variable(tf.zeros(self.output_num))
        }

        # DAE编码
        def encoder(input_):
            # feature_x = tf.nn.sigmoid(tf.matmul(x, weights['encoder_W']) + biases['encoder_b'])  # 特征层sigmoid输出
            feature_ = tf.matmul(input_, self.weights['encoder_W']) + self.biases['encoder_b']  # 特征层线性输出
            return feature_

        # dropout层
        def dropout(feature_):  # 加入dropout层，防止过拟合
            feature_dropout_ = tf.nn.dropout(feature_, self.dropout_keep_prob)
            return feature_dropout_

        # DAE解码
        def decoder(feature_dropout_):
            reconstruction_ = tf.nn.sigmoid(tf.matmul(feature_dropout_, self.weights['decoder_W'])
                                            + self.biases['decoder_b'])
            return reconstruction_

        # 特征
        self.feature = encoder(self.x)
        # dropout输出
        self.feature_dropout = dropout(self.feature)
        # 重建输入
        self.reconstruction = decoder(self.feature_dropout)
        # 均方误差损失
        self.mse = tf.reduce_mean(tf.reduce_sum(tf.pow(self.reconstruction - self.x, 2), axis=1))
        # L2正则化项
        self.L2norm = self.lamda * (tf.nn.l2_loss(self.weights['encoder_W'])
                                    + tf.nn.l2_loss(self.weights['decoder_W']))
        # 损失函数 = 均方差损失 + L2范数正则化项
        self.cost = self.mse + self.L2norm
        # 优化器
        self.optm = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

    def train(self, input_x, model_path):
        # # 先绘制计算图
        # self.draw_ComputeMap()
        # 然后训练
        dataset = DataSet(input_x, input_x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.batch_num = int(dataset.num_examples / self.batch_size)
            self.cost_history = {'Epoch': [], 'cost': []}
            print("\n******** 自编码器开始学习特征 ********")
            for epoch in range(self.epochs):
                total_cost = 0
                for _ in range(self.batch_num):
                    # 取一个batch的训练数据
                    batch_xs, _ = dataset.next_batch(self.batch_size)
                    # 加入噪声，迫使自编码器学习特征
                    batch_xs_noise = batch_xs + 0.3*np.random.randn(self.batch_size, self.input_num)
                    # 喂数据，训练
                    _, c = sess.run([self.optm, self.cost], feed_dict={self.x: batch_xs_noise})
                    total_cost += c
                    # c是这个batch中每个样本的平均误差（在这个batch内求平均）
                    # total_cost表示这次迭代所有样本的累积误差
                # cc表示每个样本的平均误差（在整个训练集的范围内求平均）
                cc = total_cost / self.batch_num
                self.cost_history['Epoch'].append(epoch)
                self.cost_history['cost'].append(cc)
                # 每n次迭代展示一次训练误差
                if epoch % 100 == 0:
                    print("Epoch:%5d     cost:%.7f" % (epoch, cc))
            print("******** 自编码器特征提取完成 ********\n")
            # 误差变化曲线
            plt.plot(self.cost_history['Epoch'], self.cost_history['cost'])
            plt.xlabel('Epcoh')
            plt.ylabel('Cost')
            plt.title('Autoencoder: Feature extraction')
            plt.show()
            # 评估模型训练效果---计算重建误差
            print("重建误差：", sess.run(self.mse, feed_dict={self.x: input_x}))
            # 保存模型
            self.model_path = model_path
            tf.train.Saver().save(sess, self.model_path)

    def get_feature(self, input_x):
        with tf.Session() as sess1:
            sess1.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess1, self.model_path)  # 读取保存的模型
            Feature = sess1.run(self.feature, feed_dict={self.x: input_x})
            return Feature


# LSTM+DAE特征提取
class LSTM_DAE(object):
    def __init__(self, input_num, hidden_num, output_num,
                 time_step, batch_size, learning_rate, lamda,
                 dropout_keep_prob, epochs):
        self.input_num = input_num  # 输入节点数
        self.hidden_num = hidden_num  # 隐层节点数
        self.output_num = output_num  # 输出节点数
        self.time_step = time_step  # 时间深度
        self.batch_size = batch_size  # batch_size必须能够被time_step整除
        self.learning_rate = learning_rate  # 学习率
        self.lamda = lamda  # 正则化参数
        self.dropout_keep_prob = dropout_keep_prob  # dropout参数
        self.epochs = epochs  # 迭代次数

    def draw_ComputeMap(self):
        # LSTM网络输入
        self.input_x = tf.placeholder(tf.float32, [None, self.input_num])
        # reshape网络输入
        self.input_xx = tf.reshape(self.input_x, [-1, self.time_step, self.input_num])
        # 定义LSTM单元
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_num)
        # 动态LSTM得到隐层输出
        self.outputs, self.final_state = tf.nn.dynamic_rnn(
            cell=self.lstm_cell,
            inputs=self.input_xx,
            initial_state=None,
            dtype=tf.float32,
            time_major=False
        )
        # 特征
        self.feature = tf.reshape(self.outputs, [-1, self.hidden_num])
        # 加入dropout层
        self.outputs_dropout = tf.nn.dropout(self.outputs, self.dropout_keep_prob)
        # 所有时刻t接上全连接层
        self.network_out = tf.layers.dense(inputs=self.outputs_dropout, units=self.output_num)
        # 重建输入
        self.reconstruction = tf.reshape(self.network_out, [-1, self.output_num])
        # 均方差损失
        self.reconstruction_error = tf.losses.mean_squared_error(
            labels=self.input_x, predictions=self.reconstruction)
        # L2正则化项
        self.L2norm = self.lamda * tf.nn.l2_loss(self.lstm_cell.weights[0])
        # 损失函数 = 均方差损失 + L2正则化项
        self.loss = self.reconstruction_error + self.L2norm
        # 优化器
        self.optm = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, input_x, model_path):
        # # 先绘制计算图
        # self.draw_ComputeMap()
        # 然后开始训练
        dataset = DataSet(input_x, input_x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.batch_num = int(dataset.num_examples / self.batch_size)
            self.cost = []
            print("\n******** LSTM&DAE开始学习特征 ********")
            for epoch in range(self.epochs):
                total_cost = 0
                for _ in range(self.batch_num):
                    # 读取一个batch的数据
                    train_x, _ = dataset.next_batch(self.batch_size)
                    # 加入噪声，迫使自编码器学习到特征
                    train_x_noise = train_x + 0.3 * np.random.randn(self.batch_size, self.input_num)
                    c, _ = sess.run([self.loss, self.optm], feed_dict={self.input_x: train_x_noise})
                    total_cost += c
                cc = total_cost / self.batch_num
                self.cost.append(cc)
                if epoch % 100 == 0:
                    print('epoch:%5d   train_loss: %.7f' % (epoch, cc))
            print("******** LSTM&DAE特征提取完成 ********\n")
            # cost曲线
            plt.plot(range(len(self.cost)), self.cost)
            plt.xlabel('Epcoh')
            plt.ylabel('Cost')
            plt.title('LSTM+DAE: Feature extraction')
            plt.show()
            # 评估模型训练效果---计算重建误差
            print('重建误差：', sess.run(self.reconstruction_error, feed_dict={self.input_x: input_x}))
            # 保存模型
            self.model_path = model_path
            tf.train.Saver().save(sess, self.model_path)

    def get_feature(self, input_x):
        with tf.Session() as sess1:
            sess1.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess1, self.model_path)  # 读取保存的模型
            Feature = sess1.run(self.feature, feed_dict={self.input_x: input_x})
            return Feature


# 神经网络分类器
class NeuralNetwork(object):
    def __init__(self, input_num, hidden_num, output_num,
                 learning_rate, epochs, decay_steps, decay_rate,
                 lamda, batch_size, MultiClass=False):
        self.input_num = input_num          # 输入节点数
        self.hidden_num = hidden_num        # 隐层节点数
        self.output_num = output_num        # 输出节点数
        self.learning_rate = learning_rate  # 学习率
        self.epochs = epochs                # 迭代次数
        self.decay_steps = decay_steps      # 学习率衰减步数
        self.decay_rate = decay_rate        # 学习率衰减率
        self.lamda = lamda                  # 正则化参数
        self.batch_size = batch_size        # 批训练大小
        self.MultiClass = MultiClass

    def draw_ComputeMap(self):
        # 当前步数
        self.global_step = tf.Variable(tf.constant(0), trainable=False)
        # 指数衰减学习率
        self.learning_rate_de = tf.train.\
            exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate)
        # NN网络输入
        self.x = tf.placeholder(tf.float32, [None, self.input_num])
        # NN网络标签
        self.y = tf.placeholder(tf.float32, [None, self.output_num])
        # 定义权重、偏置
        self.W1 = tf.Variable(tf.random_normal([self.input_num, self.hidden_num]))
        self.b1 = tf.Variable(tf.zeros(self.hidden_num))
        self.W2 = tf.Variable(tf.random_normal([self.hidden_num, self.output_num]))
        self.b2 = tf.Variable(tf.zeros(self.output_num))

        # 定义前向传播
        def forward_propagation(x_, W1_, b1_, W2_, b2_):
            hidden_out = tf.nn.sigmoid(tf.matmul(x_, W1_) + b1_)
            net_out = tf.matmul(hidden_out, W2_) + b2_
            return net_out

        # NN网络预测值
        self.pred = tf.nn.softmax(forward_propagation(self.x, self.W1, self.b1, self.W2, self.b2))
        # 交叉熵损失
        self.CrossEntropy = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), reduction_indices=1))
        # L2范数正则化项
        self.L2norm = self.lamda * tf.nn.l2_loss(self.W1) + self.lamda * tf.nn.l2_loss(self.W2)
        # 损失函数 = 交叉熵损失 + L2范数正则化项
        self.cost = self.CrossEntropy + self.L2norm
        # 优化器
        self.optm = tf.train.AdamOptimizer(self.learning_rate_de).minimize(self.cost)
        # 计算准确率Accuracy
        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self, input_x, input_y, model_path):
        # 先绘制计算图
        self.draw_ComputeMap()
        # 然后训练
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 变量初始化
            dataset = DataSet(input_x, input_y)
            batch_num = int(dataset.num_examples / self.batch_size)  # 每一次epoch迭代的batch数
            self.cost_history = {'Epoch': [], 'cost': []}  # 记录每次迭代的cost
            print("\n******** 神经网络开始训练 ********")
            for epoch in range(self.epochs):
                total_cost = 0
                for i in range(batch_num):
                    batch_xs, batch_ys = dataset.next_batch(self.batch_size)
                    _, c = sess.run([self.optm, self.cost], feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                       self.global_step: epoch})
                    total_cost += c
                self.cost_history['Epoch'].append(epoch)
                self.cost_history['cost'].append(total_cost / batch_num)
                if epoch % 20 == 0:
                    print("Epoch:%4d     cost:%.6f" % (epoch, total_cost / batch_num))
            print("******** 神经网络训练完成 ********\n")
            # 可视化误差cost
            plt.plot(self.cost_history['Epoch'], self.cost_history['cost'])
            plt.xlabel('Epcoh')
            plt.ylabel('Cost')
            plt.title('Neural Network: Classification')
            plt.show()

            if self.MultiClass:
                # 训练集准确率
                self.ac = sess.run(self.accuracy, feed_dict={self.x: input_x, self.y: input_y})
                print("训练集准确率：%.2f%s" % (100*self.ac, '%'))
            else:
                # 训练集准确率
                self.ac = sess.run(self.accuracy, feed_dict={self.x: input_x, self.y: input_y})
                print("训练集准确率：%.2f%s" % (100*self.ac, '%'))
                # 训练集F1-Score
                train_pred = sess.run(self.pred, feed_dict={self.x: input_x})
                print("训练集F1-Score：%.4f" % cap.F1_Score(np.argmax(train_pred, axis=1), np.argmax(input_y, axis=1)))
                # 训练集误报率FAR、漏检率MDR
                train_FAR, train_MDR = cap.FAR_MDR(np.argmax(train_pred, axis=1), np.argmax(input_y, axis=1))
                print("训练集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s\n" % (100 * train_FAR, '%', 100 * train_MDR, '%'))

            # 保存模型
            self.model_path = model_path
            tf.train.Saver().save(sess, self.model_path)

    def test(self, input_x, input_y):
        with tf.Session() as sess1:
            sess1.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess1, self.model_path)  # 读取保存的模型
            if self.MultiClass:
                # 测试集准确率
                self.ac = sess1.run(self.accuracy, feed_dict={self.x: input_x, self.y: input_y})
                print('测试集准确率：%.2f%s' % (100*self.ac, '%'))
            else:
                # 测试集准确率
                self.ac = sess1.run(self.accuracy, feed_dict={self.x: input_x, self.y: input_y})
                print('测试集准确率：%.2f%s' % (100*self.ac, '%'))
                # 测试集F1-Score
                test_pred = sess1.run(self.pred, feed_dict={self.x: input_x})
                print("测试集F1-Score：%.4f" % cap.F1_Score(np.argmax(test_pred, axis=1), np.argmax(input_y, axis=1)))
                # 测试集误报率FAR、漏检率MDR
                test_FAR, test_MDR = cap.FAR_MDR(np.argmax(test_pred, axis=1), np.argmax(input_y, axis=1))
                print("测试集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s\n" % (100 * test_FAR, '%', 100 * test_MDR, '%'))

    def predict(self, input_x):
        with tf.Session() as sess2:
            sess2.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess2, self.model_path)  # 读取保存的模型
            prediction = sess2.run(self.pred, feed_dict={self.x: input_x})
        return prediction


# 可视化二维特征
def visualize_feature(samples, labels, title):
    data = np.column_stack((samples, labels))
    label = ['Normal', 'Fault']  # 标注
    colors = ['cyan', 'red']  # 点的颜色
    marker = ['o', 'o']  # 点的形状
    s = [40, 40]  # 点的大小
    edgecolors = ['k', 'k']  # 边缘线颜色
    linewidths = [0.5, 0.5]  # 边缘线宽度
    alphas = [0.65, 0.65]  # 透明度
    for i in range(2):
        x1 = data[data[:, 2] == i][:, 0]
        x2 = data[data[:, 2] == i][:, 1]
        plt.scatter(x1, x2, c=colors[i], marker=marker[i], label=label[i],
                    s=s[i], linewidths=linewidths[i], edgecolors=edgecolors[i],
                    alpha=alphas[i])
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


# 得到所需数据
def Get_Data(choose_dataset, fault_class):
    """
    最新方法：
    输入：数字0~21（第0类正常样本 + 1~21故障样本）
    输出：训练集：train_x(480, 52), train_y(480, 1)
         测试集：test_x(960, 52), test_y(960, 1)
    """
    train_data = pd.read_csv('TE_Train_data.csv')
    test_data = pd.read_csv('TE_Test_data.csv')

    if choose_dataset == 'train':
        res_x = train_data.loc[train_data['V52'] == fault_class].iloc[:, :52]
        res_y = train_data.loc[train_data['V52'] == fault_class].iloc[:, 52]

    elif choose_dataset == 'test':
        start = 960 * fault_class
        end = start + 960
        res_x = test_data.iloc[start:end, :].iloc[:, :52]
        res_y = test_data.iloc[start:end, :].iloc[:, 52]

    else:
        print('Input Error !')

    return np.array(res_x), np.array(res_y)

def get_data(choose_dataset, fault_classes, null=True, outlier=False, normalize=False, onehot=True):
    """
    New way
    输入：choose_dataset: 'train'表示训练集，‘test’表示测试集
         fault_classes: 故障种类（元组形式），取值范围：1~21（共21类故障，默认加上第0类正常样本）
         null, outlier, normalize, onehot四个参数表示是否进行相应的数据预处理操作，默认均为处理（True）
         ** 修改：默认不进行异常值处理，不进行归一化处理（手动使用preprocessing包进行归一化）
    输出：data_x(样本)、data_y(标签)
         注意：对于训练集：加上前500个正常样本，后面故障类别的训练样本中不含正常类样本（480个均为此类故障）
              对于测试集：加上960个正常样本，后面故障类别的测试样本均含有正常类样本（前160个正常 + 后800个此类故障）
         *** 4.15更新：测试集不再加上前960个正产样本，输出直接就是(960, 52)和(960, 2)矩阵
    """
    pd.set_option('mode.chained_assignment', None)  # 关闭警告

    if choose_dataset == 'train':
        dataset = pd.read_csv('TE_Train_data.csv')
        data_ = dataset.loc[dataset['V52'] == 0]  # 训练集：加上前500个正常样本
        for i_ in fault_classes:
            data_ = pd.concat([data_, dataset.loc[dataset['V52'] == i_]])

    elif choose_dataset == 'test':
        dataset = pd.read_csv('TE_Test_data.csv')
        # data_ = dataset.iloc[:960, :]  # 测试集：加上前960个正常样本
        data_ = pd.DataFrame()
        for i_ in fault_classes:
            start = i_ * 960
            end = start + 960
            data_ = pd.concat([data_, dataset.iloc[start:end, :]])

    # 缺失值处理
    if null:
        data_ = data_.fillna(method='ffill')

    data_x = data_.iloc[:, :52]
    data_y = data_.iloc[:, 52]

    # 异常值处理
    if outlier:
        data_x = Outlier(data_x, 'plotbox')

    # 特征归一化
    if normalize:
        norm = preprocessing.MinMaxScaler()
        data_x = norm.fit_transform(data_x)

    # 标签one-hot编码
    if onehot:
        data_y = pd.get_dummies(data_y).values

    return np.array(data_x), np.array(data_y)

def get_data0(x, onehot=True):
    pd.set_option('mode.chained_assignment', None)  # 关闭警告
    # 导入数据
    train_data = pd.read_csv('TE_Train_data.csv')  # 训练样本
    test_data = pd.read_csv('TE_Test_data.csv')  # 测试样本

    # 缺失值处理
    train_data = train_data.fillna(method='ffill')
    test_data = test_data.fillna(method='ffill')

    # 异常值处理
    train_data = Outlier(train_data, 'plotbox')
    test_data = Outlier(test_data, 'plotbox')

    # 训练集
    start = 500 + (x - 1) * 480
    end = start + 480
    train_d0 = train_data.iloc[:500, :]
    train_dx = train_data.iloc[start:end, :]
    train_X = pd.concat([train_d0, train_dx]).iloc[:, :52]
    train_Y = pd.concat([train_d0, train_dx]).iloc[:, 52]
    # 测试集
    start = x * 960
    end = start + 960
    test_d0 = test_data.iloc[:960, :]
    test_dx = test_data.iloc[start:end, :]
    test_X = pd.concat([test_d0, test_dx]).iloc[:, :52]
    test_Y = pd.concat([test_d0, test_dx]).iloc[:, 52]

    # 样本：归一化
    train_X = preprocessing.MinMaxScaler().fit_transform(train_X)
    test_X = preprocessing.MinMaxScaler().fit_transform(test_X)

    if onehot:
        # 标签：one-hot编码
        train_Y = pd.get_dummies(train_Y)
        test_Y = pd.get_dummies(test_Y)

    return train_X, train_Y, test_X, test_Y


# 异常值处理
def Outlier(dataframe, method):
    for i in range(dataframe.shape[1]):
        data = dataframe.iloc[:, i]

        # 异常值检测
        if method == '3sigma':   # 3σ原则
            yc_bool_index = (data.mean()-3*data.std() > data) | \
                            (data.mean()+3*data.std() < data)

        if method == 'plotbox':  # 箱线图法
            QL = data.quantile(0.25)
            QU = data.quantile(0.75)
            IQR = QU - QL
            yc_bool_index = (data > QU + 1.5 * IQR) | (data < QL - 1.5 * IQR)

        if method == 'zscore':   # zscore法
            MAD = (data - data.median()).abs().median()
            # z_score = (data - data.mean()) / data.std()            # 普通zscore
            z_score = ((data - data.median()) * 0.6475 / MAD).abs()  # 增强zscore
            yc_bool_index = z_score.abs() > 3.5

        yc_index = np.arange(data.shape[0])[yc_bool_index]

        # # 可视化
        # if i % 10 == 0:  # 选择V0、V10、V20、V30、V40、V50六个展示
        #     plt.plot(data.iloc[yc_index], 'or', data, '.')
        #     plt.show()

        # 异常值处理
        data.iloc[yc_index] = data.mean()
        dataframe.iloc[:, i] = data

    return dataframe


# 定义DataSet类：包含了next_batch函数
class DataSet(object):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.epochs_completed = 0  # 完成遍历轮数
        self.index_in_epochs = 0  # 调用next_batch()函数后记住上一次位置
        self.num_examples = images.shape[0]  # 训练样本数

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        start = self.index_in_epochs
        # 初始化：将输入进行洗牌（只在最开始执行一次）
        if self.epochs_completed == 0 and start == 0 and shuffle:
            index0 = np.arange(self.num_examples)
            # print(index0)
            np.random.shuffle(index0)
            # print(index0)
            self.images = np.array(self.images)[index0]
            self.labels = np.array(self.labels)[index0]
            # print(self._images)
            # print(self._labels)
            # print("-----------------")

        # *特殊情况：取到最后，剩余样本数不足一个batch_size
        if start + batch_size > self.num_examples:
            # 先把剩余样本的取完
            self.epochs_completed += 1
            rest_num_examples = self.num_examples - start
            images_rest_part = self.images[start:self.num_examples]
            labels_rest_part = self.labels[start:self.num_examples]
            # 重新洗牌，得到新版的数据集
            if shuffle:
                index = np.arange(self.num_examples)
                np.random.shuffle(index)
                self.images = self.images[index]
                self.labels = self.labels[index]
            # 再从新的数据集中取，补全batch_size个样本
            start = 0
            self.index_in_epochs = batch_size - rest_num_examples
            end = self.index_in_epochs
            images_new_part = self.images[start:end]
            labels_new_part = self.labels[start:end]
            # 将旧的和新的拼在一起，得到特殊情况下的batch样本
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)

        # 正常情况：往后取batch_size个样本
        else:
            self.index_in_epochs += batch_size
            end = self.index_in_epochs
            return self.images[start:end], self.labels[start:end]
    # @property
    # def num_examples(self):
    #     return self.num_examples


# 保存提取到的特征
def SaveFeature(_train_feature, _test_feature):
    with open('train_feature.csv', 'w') as datafile:
        writer = csv.writer(datafile, delimiter=',')
        writer.writerows(_train_feature)
    with open('test_feature.csv', 'w') as datafile:
        writer = csv.writer(datafile, delimiter=',')
        writer.writerows(_test_feature)
    print('提取到的特征已保存为csv文件.')








