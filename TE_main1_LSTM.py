# **************** LSTM + DAE 特征提取  *****************
"""
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
fault_classes = 1,  # 选择故障类型
train_x0, _ = TE.get_data('train', fault_classes)
test_x0, _ = TE.get_data('test', fault_classes)

normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x0 = normalize.fit_transform(train_x0)
test_x0 = normalize.transform(test_x0)

train_data = TE.DataSet(train_x0, train_x0)  # 实例化DataSet

# 参数设置
input_num = 52  # 输入节点数
hidden_num = 20  # 隐层节点数
output_num = 52  # 输出节点数
time_step = 10  # 时间深度
batch_size = 10 * time_step  # batch_size必须能够被time_step整除
learning_rate = 0.1  # 学习率
lamda = 0.0001  # 正则化参数
dropout_keep_prob = 0.8  # dropout参数
epochs = 1000  # 迭代次数

# 搭建LSTM网络
input_x0 = tf.placeholder(tf.float32, [None, input_num])
input_y0 = input_x0
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
outputs_dropout = tf.nn.dropout(outputs, dropout_keep_prob)
output = tf.layers.dense(inputs=outputs_dropout, units=output_num)  # 所有时刻t接上全连接层

pred = tf.reshape(output, [-1, output_num])
true = tf.reshape(input_y, [-1, output_num])
loss = tf.losses.mean_squared_error(labels=true, predictions=pred) \
       + lamda * tf.nn.l2_loss(rnn_cell.weights[0])   # L2范数
# loss = tf.reduce_mean(tf.reduce_sum(tf.pow(pred - true, 2), axis=1)) \
#        + lamda * tf.nn.l2_loss(rnn_cell.weights[0])   # L2范数
optm = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

feature = tf.reshape(outputs, [-1, hidden_num])

# 开启会话
with tf.Session() as sess:
    # 训练
    cost = []
    sess.run(tf.global_variables_initializer())
    batch_num = int(train_x0.shape[0] / batch_size)
    for epoch in range(epochs):
        total_cost = 0
        for _ in range(batch_num):
            # 读取一个batch的数据
            train_x, _ = train_data.next_batch(batch_size)
            # 加入噪声，迫使自编码器学习到特征
            train_x_noise = train_x + 0.3*np.random.randn(batch_size, input_num)
            c, _ = sess.run([loss, optm], feed_dict={input_x0: train_x_noise})
            total_cost += c

        cc = total_cost / batch_num
        cost.append(cc)
        if epoch % 100 == 0:
            print('epoch:%d   train_loss: %.8f' % (epoch, cc))

    # cost曲线
    plt.plot(range(len(cost)), cost)
    plt.show()

    # 保存特征
    train_pred, train_feature = sess.run([pred, feature], feed_dict={input_x0: train_x0})
    test_pred, test_feature = sess.run([pred, feature], feed_dict={input_x0: test_x0})
    print("训练集的重建误差为：", np.mean(np.sum(pow(train_pred - train_x0, 2), axis=1)))
    print("测试集的重建误差为：", np.mean(np.sum(pow(test_pred - test_x0, 2), axis=1)))
    TE.SaveFeature(train_feature, test_feature)






