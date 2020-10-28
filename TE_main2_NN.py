# ********************* 神经网络分类器 **********************
# 基于TensorFlow框架

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import TE_function as TE
import cap

fault_classes = 1,  # 输入故障类别
_, train_y = TE.get_data('train', fault_classes)
_, test_y = TE.get_data('test', fault_classes)

# 读取提取到的特征
train_x = pd.read_csv('train_feature.csv', header=None).values
test_x = pd.read_csv('test_feature.csv', header=None).values


# %% 网络参数
n_input = train_x.shape[1]  # 输入节点数
n_hidden = 10  # 隐层节点数
n_output = 2  # 输出节点数
learning_rate = 0.1  # 学习率
epochs = 300  # 迭代次数
decay_steps = 100  # 学习率衰减步数
decay_rate = 0.6  # 学习率衰减率
lamda = 0  # 正则化参数


# %% 1.搭建BP神经网络模型
global_step = tf.Variable(tf.constant(0), trainable=False)  # 当前步数
learning_rate_de = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate)  # 指数衰减学习率

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

W1 = tf.Variable(tf.random_normal([n_input, n_hidden]))
b1 = tf.Variable(tf.zeros(n_hidden))
W2 = tf.Variable(tf.random_normal([n_hidden, n_output]))
b2 = tf.Variable(tf.zeros(n_output))


def forward_propagation(x_, W1_, b1_, W2_, b2_):
    hidden_out = tf.nn.sigmoid(tf.matmul(x_, W1_) + b1_)
    net_out = tf.matmul(hidden_out, W2_) + b2_
    return net_out


pred = tf.nn.softmax(forward_propagation(x, W1, b1, W2, b2))  # 以softmax概率输出（0~1）
# cost = tf.reduce_mean(pow(pred-y, 2))  # 均方差损失函数，无法收敛
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1)) \
       + lamda * tf.nn.l2_loss(W1) + lamda * tf.nn.l2_loss(W2)  # L2范数
# optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 0->2000->4000次迭代，cost：3.18->2.95->2.87
optm = tf.train.AdamOptimizer(learning_rate_de).minimize(cost)  # 2000次迭代，cost：3.05 -> 2.33（在1000代左右就收敛了）

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # 计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# %% 2.训练
batch_size = 256
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 变量初始化
    ds = TE.DataSet(train_x, train_y)  # 将训练样本赋值给ds：DataSet类的对象
    batch_num = int(train_x.shape[0] / batch_size)  # 每一次epoch迭代的batch数
    cost_history = {'Epoch': [], 'cost': []}  # 记录每次迭代的cost

    print("******** 神经网络开始训练 ********")
    for epoch in range(epochs):
        total_cost = 0
        for i in range(batch_num):
            batch_xs, batch_ys = ds.next_batch(batch_size)
            _, c = sess.run([optm, cost], feed_dict={x: batch_xs, y: batch_ys, global_step: epoch})
            total_cost += c
        cost_history['Epoch'].append(epoch)
        cost_history['cost'].append(total_cost / batch_num)
        if epoch % 100 == 0:
            print("Epoch:%4d     cost:%.6f" % (epoch, total_cost / batch_num))
    print("******** 神经网络训练完成 ********\n")

    # 可视化误差cost
    plt.plot(cost_history['Epoch'], cost_history['cost'])
    plt.xlabel('epcoh')
    plt.ylabel('cost')
    plt.title('Neural Network: Classification')
    plt.show()

    # %% 3.评估模型训练效果---计算训练集、测试集的分类准确率
    print("训练集测试准确率：%.2f%s" % (100 * sess.run(accuracy, feed_dict={x: train_x, y: train_y}), '%'))
    print("测试集测试准确率：%.2f%s" % (100 * sess.run(accuracy, feed_dict={x: test_x, y: test_y}), '%'))
