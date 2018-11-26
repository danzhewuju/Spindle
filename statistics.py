import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import os
import glob
from LogisticRegression import get_Data

from matplotlib import pyplot as plt


def statistic_info(path, save_path,key):   #文件的处理，样本的优化
    data = get_Data(path, key)
    if path == "datasets/cases":
        plt.hist(data, bins=10, density=True, color="r")
        p = os.path.join(save_path, key)
        plt.savefig(p+"_diseased")
    else:
        plt.hist(data, bins=10, density=True, color="b")
        p = os.path.join(save_path, key)
        plt.savefig(p + "_normal")
    plt.show()


def test():
    save_path = "datasets/feature"
    str1 = "Frequency"
    str2 = "Duration"
    str3 = "Amplitude"
    path1 = "datasets/cases"
    path2 = "datasets/controls"
    print(statistic_info(path1, save_path, str1))
    print(statistic_info(path1, save_path, str2))
    print(statistic_info(path1, save_path, str3))
    print(statistic_info(path2, save_path, str1))
    print(statistic_info(path2, save_path, str2))
    print(statistic_info(path2, save_path, str3))


if __name__ == '__main__':
    test()



