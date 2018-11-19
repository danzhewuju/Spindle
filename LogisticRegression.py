#!/usr/bin/python
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import os
import glob
path = "./datasets/"
feature = 13        #特征的数量


def get_distribution_Data(path, key):
    data_r = []
    for im in glob.glob(path + "/*.csv"):
        data_f = pd.read_csv(im, sep=",", skiprows=(0, 1))
        data_f = data_f[key]
        data_r.append(data_f)
    return data_r


def static_spindle_distribution(path):
    key = "Time_of_night"
    data = get_distribution_Data(path, key)
    result = []
    for tmp_d in data:
        max_n = max(tmp_d)
        data_count = np.zeros(int(max_n) + 1)
        for d in tmp_d:
            data_count[int(d)] += 1
        result.append(data_count)
    length = max(map(len, result))
    print(length)
    x_data = np.full((len(result), length), 0,np.int32 )
    for row in range(len(result)):
        length = len(result[row])                                        #统一的量化标准（全部转化为相同的维度）
        x_data[row][:length] = result[row]
    return x_data


def get_Data(path, key):
    data_r = []
    for im in glob.glob(path + "/*.csv"):
        data_f = pd.read_csv(im, sep=",", skiprows=(0, 1))
        data_f = data_f[key]
        data_r.extend(data_f)
    data_r = np.array(data_r)
    return data_r                #获取某个文件夹下面的所有数据集合的某一列


def normalization(data):
    max_d = max(data)
    min_d = min(data)
    result = []
    for d in data:
        r = (d-min_d)/(max_d-min_d)
        result.append(r)
    return result                     #归一化操作


# def person_info(data):                                    #个人信息处理
#     result = []
#     frequency = data["Frequency"]
#     duration = data["Duration"]
#     Amplitude = data["Amplitude"]
#     Time_of_night = data["Time_of_night"]
#
#     f_mean, f_std, f_min, f_max = np.mean(frequency), np.std(frequency), np.min(frequency), np.max(frequency)
#     d_mean, d_std, d_min, d_max = np.mean(duration), np.std(duration), np.min(duration), np.max(duration)
#     a_mean, a_std, a_min, a_max = np.mean(Amplitude), np.std(Amplitude), np.min(Amplitude), np.max(Amplitude)
#     t_mean, t_std, t_min, t_max = np.mean(Time_of_night), np.std(Time_of_night), np.min(Time_of_night), np.max(Time_of_night)
#     result.append(f_mean)
#     result.append(f_std)
#     result.append(f_min)
#     result.append(f_max)
#
#     result.append(d_mean)
#     result.append(d_std)
#     result.append(d_min)
#     result.append(d_max)
#
#     result.append(a_mean)
#     result.append(a_std)
#     result.append(a_min)
#     result.append(a_max)
#
#     # result.append(t_mean)
#     # result.append(t_std)
#     result.append(t_min)
#     result.append(t_max)
#     # result = normalization(result)    #归一化操作
#     # result = pd.DataFrame(result)
#     return result


def deal_info(path):   #文件的处理，样本的优化             相关的特征选择
    # paths, lable = get_path(path,flag)
    train_data = []
    labels = []
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    for inx, floder in enumerate(cate):
        for im in glob.glob(floder+"/*.csv"):
            label_temp = [0]*2   #初始化
        #     person_information = person_info(pd.read_csv(im, sep=",", skiprows=(0, 1)))
        #     train_data.append(person_information)
            label_temp[inx] = 1
            labels.append(label_temp)
            print("reading file %s" % im)
    for inx, floder in enumerate(cate):
        data1 = static_spindle_distribution(floder)
        train_data.extend(data1)
    return np.asanyarray(train_data, np.float32), np.asarray(labels)


def run():
    data, labels = deal_info(path)  # 数据处理以及标签，不同模式的数据处理
    print(data.shape[0])
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)  # 随机打乱
    data = data[arr]
    labels = labels[arr]  # 随机数据的处理

    # 数据集和验证集的划分
    ratio = 0.8
    s = np.int(num_example * ratio)
    train_data = data[:s]  # 训练集的准备
    train_labels = labels[:s]

    test_data = data[s:]
    test_labels = labels[s:]  # 测试集的准备

    x = tf.placeholder(tf.float32, shape=[None, feature])
    y = tf.placeholder(tf.float32, shape=[None, 2])

    W = tf.Variable(tf.zeros([feature, 2]))
    b = tf.Variable(tf.zeros([2]))

    actv = tf.nn.softmax(tf.matmul(x, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))

    learning_rate = 0.01
    optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
    #
    accr = tf.reduce_mean(tf.cast(pred, tf.float32))
    init = tf.global_variables_initializer()

    train_epochs = 100
    batch_size = 10
    display_step = 10

    sess = tf.Session()
    sess.run(init)
    avg_cost = 0

    for epoch in range(train_epochs):
        num_batch = np.int(train_data.__len__() / batch_size)
        x_start, y_start = 0, 0
        for i in range(num_batch):
            batch_xs = train_data[x_start:x_start + batch_size]
            batch_ys = train_labels[y_start:y_start + batch_size]
            x_start = x_start + batch_size
            y_start = y_start + batch_size
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            feeds = {x: batch_xs, y: batch_ys}
            avg_cost += sess.run(cost, feed_dict=feeds) / num_batch

        # display
        if epoch % display_step == 0:
            feeds_train = {x: batch_xs, y: batch_ys}
            feeds_test = {x: test_data, y: test_labels}
            train_acc = sess.run(accr, feed_dict=feeds_train)
            test_acc = sess.run(accr, feed_dict=feeds_test)
            print("Epoch: %03d/%03d cost:%.9f train_acc:%.3f test_acc:%.3f" % (epoch, train_epochs,
                                                                               avg_cost, train_acc, test_acc))


if __name__ == '__main__':
    run()







