# **************** LSTM 特征提取  *****************
"""
基于TesnsorFlow实现
"n vs.n"型RNN

调试记录：

"""
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
import TE_function as TE
import cap

# 建立LSTM数据集
fault_classes = 15,  # 选择故障类型
train_x0, train_y0 = TE.get_data('train', fault_classes, normalize=False)
test_x0, test_y0 = TE.get_data('test', fault_classes, normalize=False)

normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x0 = normalize.fit_transform(train_x0)
test_x0 = normalize.transform(test_x0)

train_data = TE.DataSet(train_x0, train_y0)  # 实例化DataSet

# 参数设置
time_step = 20  # 时间深度
input_num = 52  # 输入节点数
hidden_num = 100  # 隐层节点数
output_num = 2  # 输出节点数
batch_size = 64 * time_step  # batch_size必须能够被time_step整除
learning_rate = 0.01  # 学习率
epochs = 1000  # 迭代次数

# 搭建LSTM网络
input_x0 = tf.placeholder(tf.float32, [None, input_num])
input_y0 = tf.placeholder(tf.int32, [None, output_num])
input_x = tf.reshape(input_x0, [-1, time_step, input_num])
input_y = tf.reshape(input_y0, [-1, time_step, output_num])

rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_num)
outputs, final_state = tf.nn.dynamic_rnn(
    cell=rnn_cell,
    inputs=input_x,
    initial_state=None,
    dtype=tf.float32,
    time_major=False
)
output = tf.layers.dense(inputs=outputs, units=output_num)  # 所有时刻t接上全连接层

loss = tf.losses.softmax_cross_entropy(onehot_labels=input_y, logits=output)
optm = tf.train.AdamOptimizer(learning_rate).minimize(loss)

pred = tf.reshape(tf.argmax(input_y, axis=2), [-1, 1])
true = tf.reshape(tf.argmax(output, axis=2), [-1, 1])
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, true), 'float'))

# 开启会话
with tf.Session() as sess:
    # 训练
    cost = []
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        # 读取一个batch的数据
        train_x, train_y = train_data.next_batch(batch_size)

        loss_, _ = sess.run([loss, optm], feed_dict={input_x0: train_x, input_y0: train_y})
        cost.append(loss_)
        if epoch % 100 == 0:
            ac = sess.run(accuracy, feed_dict={input_x0: test_x0, input_y0: test_y0})
            print('epoch:%d   train_loss: %.8f   test_accuracy: %.2f' % (epoch, loss_, ac))
    # cost曲线
    plt.plot(range(len(cost)), cost)
    plt.show()

    # 测试
    print("训练集准确率：%.2f" % (100*sess.run(accuracy, feed_dict={input_x0: train_x0, input_y0: train_y0})), '%')
    print("测试集准确率：%.2f" % (100*sess.run(accuracy, feed_dict={input_x0: test_x0, input_y0: test_y0})), '%')










