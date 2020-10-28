# ********************* 降噪自编码器DAE 特征提取 **********************
"""
调试记录：
  1. 特征提取和分类器两个脚本的归一化处理是分开的，应该用sklearn的preprocessing包对训
练集、测试集统一处理（定义normalize = sklearn.preprocessing.MinMaxScaler(),
然后normalize随着提取到的特征（csv文件）一起传递给分类器脚本）。
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import csv
import TE_function as TE
import cap

fault_classes = 6,  # 输入故障类别
train_x, _ = TE.get_data('train', fault_classes)
test_x, _ = TE.get_data('test', fault_classes)

normalize = preprocessing.MinMaxScaler()  # 特征归一化
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)

#%% 1.配置网络参数
n_input = 52  # 输入节点数
n_hidden = 20  # 隐层节点数
n_output = 52  # 输出节点数
learning_rate = 0.9  # 学习率
dropout_keep_prob = 0.8  # dropout参数
epochs = 500  # 迭代次数
lamda = 0.001  # 正则化参数


# %% 2.搭建AE网络模型
x = tf.placeholder(tf.float32, [None, n_input])
weights = {
    'encoder_W': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'decoder_W': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
    'encoder_b': tf.Variable(tf.zeros(n_hidden)),
    'decoder_b': tf.Variable(tf.zeros(n_output))
}

def encoder(input_):
    # feature_x = tf.nn.sigmoid(tf.matmul(x, weights['encoder_W']) + biases['encoder_b'])  # 特征层sigmoid输出
    feature_ = tf.matmul(input_, weights['encoder_W']) + biases['encoder_b']  # 特征层线性输出
    return feature_

def dropout(feature_):  # 加入dropout层，防止过拟合
    feature_dropout_ = tf.nn.dropout(feature_, dropout_keep_prob)
    return feature_dropout_

def decoder(feature_dropout_):
    reconstruction_ = tf.nn.sigmoid(tf.matmul(feature_dropout_, weights['decoder_W']) + biases['decoder_b'])
    return reconstruction_


feature = encoder(x)  # 特征
feature_dropout = dropout(feature)  # dropout输出
reconstruction = decoder(feature_dropout)  # 重建输入

# 损失函数：先对特征维度（列）的误差求和，再对样本维度（行）的误差求平均
# 相当于回归问题，用均方误差损失函数
cost = tf.reduce_mean(tf.reduce_sum(tf.pow(reconstruction - x, 2), axis=1)) \
       + lamda * (tf.nn.l2_loss(weights['encoder_W']) + tf.nn.l2_loss(weights['decoder_W']))  # L2范数
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# %% 3.训练
batch_size = 256
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ds = TE.DataSet(train_x, train_x)
    batch_num = int(train_x.shape[0] / batch_size)
    cost_history = {'Epoch': [], 'cost': []}

    print("******** 自编码器开始学习特征 ********")
    for epoch in range(epochs):
        total_cost = 0
        for _ in range(batch_num):
            # 取一个batch的训练数据
            batch_xs, _ = ds.next_batch(batch_size)
            # 加入噪声，迫使自编码器学习特征
            batch_xs_noise = batch_xs + 0.3*np.random.randn(batch_size, n_input)
            # 喂数据，训练
            _, c = sess.run([optm, cost], feed_dict={x: batch_xs_noise})
            total_cost += c
            # c是这个batch中每个样本的平均误差（在这个batch内求平均）
            # total_cost表示这次迭代所有样本的累积误差
        # cc表示每个样本的平均误差（在整个训练集的范围内求平均）
        cc = total_cost / batch_num
        cost_history['Epoch'].append(epoch)
        cost_history['cost'].append(cc)
        # 每n次迭代展示一次训练误差
        if epoch % 10 == 0:
            print("Epoch:%5d     cost:%.6f" % (epoch, cc))
    print("******** 自编码器特征提取完成 ********\n")

    # 误差变化曲线
    plt.plot(cost_history['Epoch'], cost_history['cost'])
    plt.xlabel('epcoh')
    plt.ylabel('cost')
    plt.title('Autoencoder: Feature extraction')
    plt.show()

    # 保存模型
    model_path = "D:\\Python Codes\\TE\\SaveTE_Feature/SaveTE_Feature.ckpt"
    tf.train.Saver().save(sess, model_path)


# %% 4.评估模型训练效果---计算训练集、测试集的重建误差
with tf.Session() as sess1:
    sess1.run(tf.global_variables_initializer())
    model_path = "D:\\Python Codes\\TE\\SaveTE_Feature/SaveTE_Feature.ckpt"
    tf.train.Saver().restore(sess1, model_path)  # 读取保存的模型

    train_pred = sess1.run(reconstruction, feed_dict={x: train_x})
    test_pred = sess1.run(reconstruction, feed_dict={x: test_x})
    print("训练集的重建误差为：", np.mean(np.sum(pow(train_pred - train_x, 2), axis=1)))
    print("测试集的重建误差为：", np.mean(np.sum(pow(test_pred - test_x, 2), axis=1)))


# %% 5.保存提取到的特征，写入csv文件
    train_feature = sess1.run(feature, feed_dict={x: train_x})  # 训练集的特征
    test_feature = sess1.run(feature, feed_dict={x: test_x})    # 测试集的特征
    TE.SaveFeature(train_feature, test_feature)





