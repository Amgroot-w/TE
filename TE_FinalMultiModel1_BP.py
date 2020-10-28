# ********** Final MultiClass Model -- 1.BP多分类 *****************
"""
调试记录:

"""
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
import TE_function as TE
import csv
import cap
import time
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
start_time = time.time()

# 导入数据
# fault_names = ['Normal', 'Fault1', 'Fault2', 'Fault6', 'Fault14', 'Fault18']
fault_names = ['正常', '故障1', '故障2', '故障6', '故障14', '故障18']

fault_classes = 1, 2, 6, 14, 18
train_x, train_y = TE.get_data('train', fault_classes)
fault_classes = 0, 1, 2, 6, 14, 18
test_x, test_y = TE.get_data('test', fault_classes)
# 特征归一化
normalize = preprocessing.MinMaxScaler()
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)

# 配置网络参数
n_input = 52
n_hidden = 20
n_output = train_y.shape[1]
learning_rate = 0.08
epochs = 2000
batch_size = 256

# 搭建网络模型
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])

W1 = tf.Variable(tf.random_normal([n_input, n_hidden]))
b1 = tf.Variable(tf.zeros(n_hidden))
W2 = tf.Variable(tf.random_normal([n_hidden, n_output]))
b2 = tf.Variable(tf.zeros(n_output))

def BP(_x, _W1, _b1, _W2, _b2):
    hidden_out = tf.nn.sigmoid(tf.matmul(_x, _W1) + _b1)
    net_out = tf.nn.sigmoid(tf.matmul(hidden_out, _W2) + _b2)
    return net_out


pred = tf.nn.softmax(BP(x, W1, b1, W2, b2))  # 以概率输出（0~1）
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))  # 交叉熵损失函数
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # 绘制计算图
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 变量初始化
    ds = TE.DataSet(train_x, train_y)  # 将训练样本赋值给ds：DataSet类的对象
    batch_num = int(train_x.shape[0] / batch_size)  # 每一次epoch迭代的batch数
    cost_history = {'Epoch': [], 'cost': []}  # 记录每次迭代的cost

    for epoch in range(epochs):
        total_cost = 0
        for i in range(batch_num):
            batch_xs, batch_ys = ds.next_batch(batch_size)
            _, c = sess.run([optm, cost], feed_dict={x: batch_xs, y: batch_ys})
            total_cost += c

        cost_history['Epoch'].append(epoch)
        cost_history['cost'].append(total_cost / batch_num)

    # 可视化误差cost
    plt.plot(cost_history['Epoch'], cost_history['cost'])
    plt.show()

    train_accuracy = sess.run(accuracy, feed_dict={x: train_x, y: train_y})
    test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    print("训练集测试准确率：%.2f%s" % (100 * train_accuracy, '%'))
    print("测试集测试准确率：%.2f%s" % (100 * test_accuracy, '%'))

    # %% 测试集的故障实时监控图
    test_pred = sess.run(pred, feed_dict={x: test_x})
    count_sum = 0
    for k in range(len(fault_classes)):
        start = 960 * k
        end = start + 960
        if k == 0:
            test_normal_prob = np.argmax(test_pred[start:end, :], axis=1).reshape([-1, 1])
            count = 960 - np.sum(test_normal_prob == 0)
            print('%s：误报个数：%d，误报率：%.2f%s' % (fault_names[k], count, (100 * count / 960), '%'))
        else:
            test_fault_prob = test_pred[start:end, k].reshape([-1, 1])
            TE.test_plot(fault_names[k], test_fault_prob)

end_time = time.time()
print('运行时间：%.3f s' % (end_time - start_time))



