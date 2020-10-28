# ********************* Final Model -- 1.两层BP直接分类 **********************
"""
NN网络结构：三层BP神经网络：52个输入节点、30个隐层节点、22个输出结点

** 作为对比试验：将最终模型与最简单的BP模型比较;
** 结果保存为csv文件，在TE_plot.py中运行查看;
** 测试集准确率低于80%的故障类别：3, 9, 10, 11, 13, 14, 15, 16, 19, 20, 21

"""
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
import TE_function as TE
import csv
import cap

def NN(fault_classes):
    """
    将单故障诊断的代码定义为函数，方便研究每一类故障的分类效果
    输入故障类别即可得到bp分类器的结果：训练集准确率、测试集准确率
    """
    # 导入数据
    # fault_classes = fault_class,  # 输入故障类别
    train_x, train_y = TE.get_data('train', fault_classes)
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
    disp_step = 100
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
    # cost = tf.reduce_mean(pow(pred-y, 2))  # 均方差损失函数，无法收敛
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))  # 交叉熵损失函数，能够收敛
    optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # 0->2000->4000次迭代，cost：3.18->2.95->2.87
    # optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # 2000次迭代，cost：3.05 -> 2.33（在1000代左右就收敛了）
    # optm = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  # 绘制计算图
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 变量初始化
        ds = TE.DataSet(train_x, train_y)  # 将训练样本赋值给ds：DataSet类的对象
        batch_num = int(train_x.shape[0] / batch_size)  # 每一次epoch迭代的batch数
        cost_history = {'Epoch': [], 'cost': []}  # 记录每次迭代的cost

        # print("******** 开始训练 ********")
        for epoch in range(epochs):
            total_cost = 0
            for i in range(batch_num):
                batch_xs, batch_ys = ds.next_batch(batch_size)
                # batch_xs_noise = batch_xs + 0.3*np.random.randn(batch_size, n_input)
                _, c = sess.run([optm, cost], feed_dict={x: batch_xs, y: batch_ys})
                total_cost += c

            cost_history['Epoch'].append(epoch)
            cost_history['cost'].append(total_cost / batch_num)

        #     if epoch % disp_step == 0:
        #         print("Epoch:%04d  cost:%.6f" % (epoch, total_cost / batch_num))
        # print("******** 训练完成 ********")

        # 可视化误差cost
        plt.plot(cost_history['Epoch'], cost_history['cost'])
        plt.show()

        # # 保存模型
        # model_path = "D:\\Python Codes\\TE\\SaveTE/TE_train_model.ckpt"
        # tf.train.Saver().save(sess, model_path)

    # 测试
    # with tf.Session() as sess1:  # 开启会话
    #     sess1.run(tf.global_variables_initializer())
    #     model_path = "D:\\Python Codes\\TE\\SaveTE/TE_train_model.ckpt"
    #     tf.train.Saver().restore(sess1, model_path)

        train_accuracy = sess.run(accuracy, feed_dict={x: train_x, y: train_y})
        test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        print("训练集测试准确率：%.2f%s" % (100*train_accuracy, '%'))
        print("测试集测试准确率：%.2f%s" % (100*test_accuracy, '%'))

        train_pred = sess.run(pred, feed_dict={x: train_x})
        test_pred = sess.run(pred, feed_dict={x: test_x})
        print("训练集F1-Score：%.4f" % cap.F1_Score(np.argmax(train_pred, axis=1), np.argmax(train_y, axis=1)))
        print("测试集F1-Score：%.4f" % cap.F1_Score(np.argmax(test_pred, axis=1), np.argmax(test_y, axis=1)))

        train_FAR, train_MDR = cap.FAR_MDR(np.argmax(train_pred, axis=1), np.argmax(train_y, axis=1))
        test_FAR, test_MDR = cap.FAR_MDR(np.argmax(test_pred, axis=1), np.argmax(test_y, axis=1))
        print("训练集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s" % (100 * train_FAR, '%', 100 * train_MDR, '%'))
        print("测试集：误报率FAR：%.2f%s，漏检率MDR：%.2f%s" % (100 * test_FAR, '%', 100 * test_MDR, '%'))

        # 测试集的故障实时监控图、故障检出点、漏检个数
        test_fault_prob = test_pred[:, 1]
        TE.test_plot(test_fault_prob)

    return train_accuracy, test_accuracy


def compute_accuracy_of_all_faults():
    # 迭代五次求平均
    train_ac = np.zeros([21, 5])
    test_ac = np.zeros([21, 5])
    for i in range(21):
        # 测试每一类故障
        for j in range(5):
            ac1, ac2 = NN(i+1)  # 调用
            train_ac[i][j] = ac1
            test_ac[i][j] = ac2
            print('第%d类故障，第%d次迭代完成！' % (i, j))
    train_ac = np.column_stack((np.arange(1, 22), train_ac))
    test_ac = np.column_stack((np.arange(1, 22), test_ac))
    # 保存结果
    with open('accuracy_train.csv', 'w') as datafile:
        writer = csv.writer(datafile, delimiter=',')
        writer.writerows(train_ac)
    with open('accuracy_test.csv', 'w') as datafile:
        writer = csv.writer(datafile, delimiter=',')
        writer.writerows(test_ac)
    print('\n所有21类故障的五次迭代结果（训练集、测试集准确率）已保存为csv文件.')


if __name__ == '__main__':
    # compute_accuracy_of_all_faults()
    fuault_classes = 1, 2, 4, 6, 7, 18
    NN(fuault_classes)







