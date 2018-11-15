#!/usr/bin/python
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import os
import glob
path = "datasets/controls"


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


print(static_spindle_distribution(path).shape)






