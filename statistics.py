import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import os
import glob
import LogisticRegression

from matplotlib import pyplot as plt

# 参数依次为list,抬头,X轴标签,Y轴标签,XY轴的范围


def draw_hist(data, min_a,max_b, Xlabel, Ylabel):
    plt.hist(data, bins=10, density=True, alpha=0.5, color='b', histtype='stepfilled',
         edgecolor='none')
    # plt.title('Spindles data histogram')
    # plt.xlabel(Xlabel)
    # plt.ylabel(Ylabel)
    # save_path = os.path.join("./datasets/feature", Xlabel)
    # plt.savefig(save_path)
    plt.show()


def statistic_info(path, key):   #文件的处理，样本的优化
    count = 10
    data = []
    for im in glob.glob(path + "/*.csv"):
        data_f = pd.read_csv(im, sep=",", skiprows=(0, 1))
        data_f = data_f[key]
        data.extend(data_f)
    max_t = max(data)
    min_t = min(data)
    step = (max_t-min_t)/count
    result = [0]*count
    x_list = [0]*count
    for i in range(count):
        x_list[i] = i*step + min_t
    total_counts = data.__len__()
    for d in data:
        n = int((d-min_t) // step)
        result[n] += 1
    # for index in range(result.__len__()):
    #     result[index] = result[index]/total_counts
    draw_hist(data, min_t, max_t, key, "count")

    return result


str1 = "Frequency"
str2 = "Duration"
str3 = "Amplitude"
path1 = "datasets/cases"
path2 = "datasets/controls"
print(statistic_info(path1, str1))
print(statistic_info(path1, str2))
print(statistic_info(path1, str3))
print(statistic_info(path2, str1))
print(statistic_info(path2, str2))
print(statistic_info(path2, str3))
