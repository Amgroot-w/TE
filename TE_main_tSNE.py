# **************** SAE特征提取+ t-SNE可视化 + DBSCAN聚类 *****************
# 待完成： SAE的优化， DBSCAN编写

import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import manifold

import TE_function as TE

#%% 选择样本
# 1类正常，11类故障：1,2,4,6,7,11,13,14,17,19,20
fault_classes = 1, 2, 4, 6, 7, 11, 13, 14, 17, 19, 20  # 故障类别
train_x, train_y = TE.get_data('train', fault_classes, onehot=False)


#%% SAE特征提取
n_input = 52  # 输入节点数
n_hidden = 20  # 隐层节点数
n_output = 52  # 输出节点数
learning_rate = 0.5  # 学习率
dropout_keep_prob = 1  # dropout参数
epochs = 200  # 迭代次数
lamda = 0.0001  # 正则化参数

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

cost = tf.reduce_mean(tf.reduce_sum(tf.pow(reconstruction - x, 2), axis=1)) \
       + lamda * (tf.nn.l2_loss(weights['encoder_W']) + tf.nn.l2_loss(weights['decoder_W']))  # L2范数
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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
            batch_xs, batch_ys = ds.next_batch(batch_size)
            batch_xs_noise = batch_xs + 0.3*np.random.randn(batch_size, n_input)
            _, c = sess.run([optm, cost], feed_dict={x: batch_xs_noise})
            total_cost += c
        cc = total_cost / batch_num
        cost_history['Epoch'].append(epoch)
        cost_history['cost'].append(cc)
        if epoch % 10 == 0:
            print("Epoch:%5d     cost:%.6f" % (epoch, cc))
    print("******** 自编码器特征提取完成 ********\n")

    plt.plot(cost_history['Epoch'], cost_history['cost'])
    plt.xlabel('epcoh')
    plt.ylabel('cost')
    plt.title('Autoencoder: Feature extraction')
    plt.show()

    # 得到训练集的特征
    train_feature = sess.run(feature, feed_dict={x: train_x})

#%% t-SNE可视化
print('t-SNE拟合... ...')
tsne = manifold.TSNE(n_components=2)
train_tsne = tsne.fit_transform(train_feature)
print('t-SNE拟合完成.')

plt.scatter(train_tsne[:, 0], train_tsne[:, 1], s=2)
plt.show()
plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_y, s=2)
plt.show()











