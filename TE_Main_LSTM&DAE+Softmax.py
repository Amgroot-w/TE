# %% ********************* LSTM & DAE + Sofmax **********************
"""
调试记录：

"""
# %% 导入包
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing, manifold
import TE_function as TE
import cap


# %% 准备数据集
"""
对于每一类故障: (500 + 480) = 980 个训练样本
             (960 + 160+800) = 1920 个测试样本
将980个训练样本分为两部分： 无监督预训练（学习特征）: (400个正常 + 360个故障) = 760个样本
                       有监督训练微调（分类）: (100个正常 + 120个故障) = 220个样本
"""
# 输入故障类别（范围：1~21）
fault_classes = 1,
train_x, train_y = TE.get_data('train', fault_classes)
test_x, test_y = TE.get_data('test', fault_classes)

# 特征归一化
normalize = preprocessing.MinMaxScaler()
train_x = normalize.fit_transform(train_x)
test_x = normalize.transform(test_x)

# 分配训练集
train_data = TE.DataSet(train_x, train_y)  # 全部训练数据集

pre_train_x = np.row_stack((train_x[:400, :], train_x[500:860, :]))
pre_train_data = TE.DataSet(pre_train_x, pre_train_x)  # 预训练数据集

tune_train_x = np.row_stack((train_x[400:500, :], train_x[860:980, :]))
tune_train_y = np.row_stack((train_y[400:500, :], train_y[860:980, :]))
tune_train_data = TE.DataSet(tune_train_x, tune_train_y)  # 微调数据集


# %% 参数设置
"""
选择网络结构：
    取值为True表示： 先预训练学习特征，再训练分类器
    取值为False表示： 不预训练，直接训练分类器
"""
pre_train_option = True
"""
选择运行模式：
    取值为True表示： 训练模型
    取值为False表示： 反复运行得到结果
"""
RunBestModelAgain = False

# LSTM & DAE网络参数
n_input1 = 52                 # 输入节点数
n_hidden1 = 20                # 隐层节点数
n_output1 = 52                # 输出节点数
time_step = 10                # 时间深度
batch_size1 = 10*time_step    # 批处理大小
learning_rate1 = 0.01         # 学习率
lamda1 = 0.01                 # 正则化参数
dropout_keep_prob1 = 0.8      # dropout参数
epochs1 = 1000                # 迭代次数

# 分类器网络参数
n_input2 = n_hidden1          # 输入节点数
n_hidden2 = 8                 # 隐层节点数
n_output2 = 2                 # 输出节点数
learning_rate2 = 0.04         # 学习率
epochs2 = 4000                # 迭代次数
lamda2 = 0.002                # 正则化参数
batch_size2 = 10*time_step    # 批处理大小
decay_steps = 100             # 学习率衰减步数
decay_rate = 0.65             # 学习率衰减率


# %% DAE计算图
input_x = tf.placeholder(tf.float32, [None, n_input1])
input_xx = tf.reshape(input_x, [-1, time_step, n_input1])

rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden1)
outputs, final_state = tf.nn.dynamic_rnn(
    cell=rnn_cell,
    inputs=input_xx,
    initial_state=None,
    dtype=tf.float32,
    time_major=False
)

feature = tf.reshape(outputs, [-1, n_hidden1])  # 特征
outputs_dropout = tf.nn.dropout(outputs, dropout_keep_prob1)  # dropout输出
output = tf.layers.dense(inputs=outputs_dropout, units=n_input1)  # 所有时刻t接上全连接层
reconstruction = tf.reshape(output, [-1, n_input1])  # 重建输入

reconstruct_error = tf.losses.mean_squared_error(labels=input_x, predictions=reconstruction)
regularization1 = lamda1 * tf.nn.l2_loss(rnn_cell.weights[0])
cost1 = reconstruct_error + regularization1  # 损失函数 = 均方差损失 + 正则化项
optm1 = tf.train.RMSPropOptimizer(learning_rate1).minimize(cost1)


# %% 分类器计算图
global_step = tf.Variable(tf.constant(0), trainable=False)  # 当前步数
learning_rate_de = tf.train.exponential_decay(learning_rate2, global_step, decay_steps, decay_rate)  # 指数衰减学习率

true_y = tf.placeholder(tf.float32, [None, n_output2])

W1 = tf.Variable(tf.random_normal([n_input2, n_hidden2]))
b1 = tf.Variable(tf.zeros(n_hidden2))
W2 = tf.Variable(tf.random_normal([n_hidden2, n_output2]))
b2 = tf.Variable(tf.zeros(n_output2))

hidden_out = tf.nn.relu(tf.matmul(feature, W1) + b1)
pred_y = tf.nn.softmax(tf.matmul(hidden_out, W2) + b2)

regularization2 = lamda2*tf.nn.l2_loss(rnn_cell.weights[0]) \
                  + lamda2*tf.nn.l2_loss(W1) + lamda2*tf.nn.l2_loss(W2)  # 正则化项
classify_error = tf.reduce_mean(-tf.reduce_sum(true_y * tf.log(pred_y), reduction_indices=1))  # 交叉熵损失
cost2 = classify_error + regularization2  # 损失函数 = 交叉熵损失 + 正则化项2(包含自编码器中编码部分的权重)
optm2 = tf.train.GradientDescentOptimizer(learning_rate2).minimize(cost2)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(true_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# %% 训练模型
if not RunBestModelAgain:
    # 预训练--无监督学习特征
    if pre_train_option:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_num1 = int(pre_train_data.num_examples / batch_size1)
            cost_history = {'Epoch': [], 'cost': []}

            print("******** 自编码器开始学习特征 ********")
            for epoch in range(epochs1):
                total_cost = 0
                for _ in range(batch_num1):
                    # 取一个batch的训练数据
                    batch_xs, _ = pre_train_data.next_batch(batch_size1)
                    # 加入噪声，迫使自编码器学习特征
                    batch_xs_noise = batch_xs + 0.3*np.random.randn(batch_size1, n_input1)
                    # 喂数据，训练
                    _, c = sess.run([optm1, cost1], feed_dict={input_x: batch_xs_noise})
                    total_cost += c
                    # c是这个batch中每个样本的平均误差（在这个batch内求平均）
                    # total_cost表示这次迭代所有样本的累积误差
                # cc表示每个样本的平均误差（在整个训练集的范围内求平均）
                cc = total_cost / batch_num1
                cost_history['Epoch'].append(epoch)
                cost_history['cost'].append(cc)
                # 每n次迭代展示一次训练误差
                if epoch % 10 == 0:
                    print("Epoch:%5d\tcost:%.6f" % (epoch, cc))
            print("******** 自编码器特征提取完成 ********\n")

            # 误差变化曲线
            plt.plot(cost_history['Epoch'], cost_history['cost'])
            plt.xlabel('epcoh')
            plt.ylabel('cost')
            plt.title('Autoencoder: Feature extraction')
            plt.show()

            # 评估模型训练效果---计算训练集、测试集的重建误差
            # 算的是整个训练集(980，52)、整个测试集(1920，52)的重建误差
            print("训练集的重建误差为：", sess.run(reconstruct_error, feed_dict={input_x: train_x}))
            print("测试集的重建误差为：", sess.run(reconstruct_error, feed_dict={input_x: test_x}))

            # 保存模型
            model_path = "D:\\Python Codes\\TE\\TE_SaveMainModel_LSTM&DAE1/PreModel.ckpt"
            tf.train.Saver().save(sess, model_path)

            train_feature = sess.run(feature, feed_dict={input_x: train_x})  # 训练集的特征
            test_feature = sess.run(feature, feed_dict={input_x: test_x})  # 测试集的特征
            TE.SaveFeature(train_feature, test_feature)

            # *********
            # t-SNE二维可视化高维特征
            train_feature = sess.run(feature, feed_dict={input_x: train_x})
            test_feature = sess.run(feature, feed_dict={input_x: test_x})
            print('训练集t-SNE拟合... ...')
            tsne1 = manifold.TSNE(n_components=2)
            train_tsne = tsne1.fit_transform(train_feature)
            print('测试集t-SNE拟合... ...')
            tsne2 = manifold.TSNE(n_components=2)
            test_tsne = tsne2.fit_transform(test_feature)
            print('拟合完成.')
            plt.subplot(121)
            TE.visualize_feature(fault_classes[0], train_tsne, train_y)
            plt.subplot(122)
            TE.visualize_feature(fault_classes[0], test_tsne, test_y)
            plt.show()
            # *********

    # 分类器微调
    with tf.Session() as sess1:
        sess1.run(tf.global_variables_initializer())

        if pre_train_option:
            model_path = "D:\\Python Codes\\TE\\TE_SaveMainModel_LSTM&DAE1/PreModel.ckpt"
            tf.train.Saver().restore(sess1, model_path)  # 读取保存的模型

        batch_num2 = int(tune_train_data.num_examples / batch_size2)  # 每一次epoch迭代的batch数
        cost_history = {'Epoch': [], 'cost': []}  # 记录每次迭代的cost

        print("******** 神经网络开始训练 ********")
        for epoch in range(epochs2):
            total_cost = 0
            for i in range(batch_num2):
                batch_xs, batch_ys = tune_train_data.next_batch(batch_size2)
                _, c = sess1.run([optm2, cost2], feed_dict={input_x: batch_xs, true_y: batch_ys, global_step: epoch})
                total_cost += c
            cost_history['Epoch'].append(epoch)
            cost_history['cost'].append(total_cost / batch_num2)
            if epoch % 20 == 0:
                print("Epoch:%4d     cost:%.6f" % (epoch, total_cost / batch_num2))
        print("******** 神经网络训练完成 ********\n")

        # 可视化误差cost
        plt.plot(cost_history['Epoch'], cost_history['cost'])
        plt.xlabel('epcoh')
        plt.ylabel('cost')
        plt.title('Neural Network: Classification')
        plt.show()

        # 保存最终的完整模型
        model_path = "D:\\Python Codes\\TE\\TE_SaveMainModel_LSTM&DAE2/FinalModel.ckpt"
        tf.train.Saver().save(sess1, model_path)

        # %% 评估模型训练效果---计算训练集、测试集的：分类准确率、F1-Score
        print("训练集准确率：%.2f%s" % (100 * sess1.run(accuracy, feed_dict={input_x: train_x, true_y: train_y}), '%'))
        print("测试集准确率：%.2f%s" % (100 * sess1.run(accuracy, feed_dict={input_x: test_x, true_y: test_y}), '%'))

        train_pred = sess1.run(pred_y, feed_dict={input_x: train_x})
        test_pred = sess1.run(pred_y, feed_dict={input_x: test_x})
        print("训练集F1-Score：%.4f" % cap.F1_Score(np.argmax(train_pred, axis=1), np.argmax(train_y, axis=1)))
        print("测试集F1-Score：%.4f" % cap.F1_Score(np.argmax(test_pred, axis=1), np.argmax(test_y, axis=1)))

        train_FAR, train_MDR = cap.FAR_MDR(np.argmax(train_pred, axis=1), np.argmax(train_y, axis=1))
        test_FAR, test_MDR = cap.FAR_MDR(np.argmax(test_pred, axis=1), np.argmax(test_y, axis=1))
        print("训练集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s" % (100*train_FAR, '%', 100*train_MDR, '%'))
        print("测试集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s" % (100*test_FAR, '%', 100*test_MDR, '%'))

        # t-SNE二维可视化高维特征
        train_feature = sess1.run(feature, feed_dict={input_x: train_x})
        test_feature = sess1.run(feature, feed_dict={input_x: test_x})
        print('训练集t-SNE拟合... ...')
        tsne1 = manifold.TSNE(n_components=2)
        train_tsne = tsne1.fit_transform(train_feature)
        print('测试集t-SNE拟合... ...')
        tsne2 = manifold.TSNE(n_components=2)
        test_tsne = tsne2.fit_transform(test_feature)
        print('拟合完成.')
        plt.subplot(121)
        TE.visualize_feature(fault_classes[0], train_tsne, train_y)
        plt.subplot(122)
        TE.visualize_feature(fault_classes[0], test_tsne, test_y)
        plt.show()

        # 测试集的故障实时监控图
        test_fault_prob = test_pred[:, 1]
        plt.plot(np.arange(0, 960), 0.5*np.ones(960), 'k--', label='Threshold: 50%')
        plt.plot(np.arange(0, 160), test_fault_prob[:160], 'b', label='Feature State')
        plt.plot(np.arange(160, 960), test_fault_prob[160:960], 'r', label='Fault State')
        plt.title('Real-time Monitoring')
        plt.xlabel('Test samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # 测试集故障检出点、漏检个数
        count = 0
        for i in range(len(test_fault_prob)):
            if test_fault_prob[i] > 0.5:
                break
            else:
                count += 1
        print('测试集的故障检出点：%d，漏检个数：%d' % (count, count-160))

# %% 读取上一步骤保存的最优模型，运行出结果
if RunBestModelAgain:
    with tf.Session() as sess2:
        sess2.run(tf.global_variables_initializer())
        model_path = "D:\\Python Codes\\TE\\TE_SaveMainModel_TheBestModel/FinalModel.ckpt"
        tf.train.Saver().restore(sess2, model_path)  # 读取保存的模型

        # 评估模型训练效果---计算训练集、测试集的：分类准确率、F1-Score
        print("训练集准确率：%.2f%s" % (100 * sess2.run(accuracy, feed_dict={input_x: train_x, true_y: train_y}), '%'))
        print("测试集准确率：%.2f%s" % (100 * sess2.run(accuracy, feed_dict={input_x: test_x, true_y: test_y}), '%'))

        train_pred = sess2.run(pred_y, feed_dict={input_x: train_x})
        test_pred = sess2.run(pred_y, feed_dict={input_x: test_x})
        print("训练集F1-Score：%.4f" % cap.F1_Score(np.argmax(train_pred, axis=1), np.argmax(train_y, axis=1)))
        print("测试集F1-Score：%.4f" % cap.F1_Score(np.argmax(test_pred, axis=1), np.argmax(test_y, axis=1)))

        train_FAR, train_MDR = cap.FAR_MDR(np.argmax(train_pred, axis=1), np.argmax(train_y, axis=1))
        test_FAR, test_MDR = cap.FAR_MDR(np.argmax(test_pred, axis=1), np.argmax(test_y, axis=1))
        print("训练集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s" % (100*train_FAR, '%', 100*train_MDR, '%'))
        print("测试集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s" % (100*test_FAR, '%', 100*test_MDR, '%'))

        # t-SNE二维可视化高维特征
        train_feature = sess2.run(feature, feed_dict={input_x: train_x})
        test_feature = sess2.run(feature, feed_dict={input_x: test_x})
        print('训练集t-SNE拟合... ...')
        tsne1 = manifold.TSNE(n_components=2)
        train_tsne = tsne1.fit_transform(train_feature)
        print('测试集t-SNE拟合... ...')
        tsne2 = manifold.TSNE(n_components=2)
        test_tsne = tsne2.fit_transform(test_feature)
        print('拟合完成.')
        plt.subplot(121)
        TE.visualize_feature(fault_classes[0], train_tsne, train_y)
        plt.subplot(122)
        TE.visualize_feature(fault_classes[0], test_tsne, test_y)
        plt.show()

        # 测试集的故障实时监控图
        test_fault_prob = test_pred[:, 1]
        plt.plot(np.arange(0, 960), 0.5*np.ones(960), 'k--', label='Threshold: 50%')
        plt.plot(np.arange(0, 160), test_fault_prob[:160], 'b', label='Feature State')
        plt.plot(np.arange(160, 960), test_fault_prob[160:960], 'r', label='Fault State')
        plt.title('Real-time Monitoring')
        plt.xlabel('Test samples')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()



